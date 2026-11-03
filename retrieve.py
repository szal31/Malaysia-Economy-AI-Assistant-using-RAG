"""Three ways to find the passages that answer a question: BM25 keywords, Gemini embeddings, and both.

All three return row numbers into the passage table, best first.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from reports import SourceChanged

INDEX = Path(__file__).parent / "index"


def tokens(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, chunks, emb):
        self.chunks, self.emb = chunks.reset_index(drop=True), emb
        self._bm25 = BM25Okapi([tokens(t) for t in self.chunks.text])

    def bm25(self, q, k):
        return np.argsort(-self._bm25.get_scores(tokens(q)), kind="stable")[:k].tolist()

    def dense(self, qvec, k):
        return np.argsort(-(self.emb @ qvec), kind="stable")[:k].tolist()

    def hybrid(self, q, qvec, k, rrf_k=60):
        """Reciprocal rank fusion: each method votes 1/(rrf_k + rank) for its top 50."""
        score = {}
        for ranked in (self.bm25(q, 50), self.dense(qvec, 50)):
            for rank, i in enumerate(ranked):
                score[i] = score.get(i, 0) + 1 / (rrf_k + rank + 1)
        return sorted(score, key=lambda i: -score[i])[:k]


def passages(r, question, method, titles, k):
    """The top-k passages for a question, numbered for citation. Falls back to BM25 if Gemini is down."""
    import llm
    try:
        qvec = llm.embed([question], "RETRIEVAL_QUERY")[0]
        rows = r.bm25(question, k) if method == "bm25" else \
            r.dense(qvec, k) if method == "dense" else r.hybrid(question, qvec, k)
    except llm.Unavailable:  # keyword search needs no API call
        rows = r.bm25(question, k)
    return [{"n": n, "doc_id": c.doc_id, "title": titles[c.doc_id], "page": int(c.page), "text": c.text}
            for n, c in enumerate(r.chunks.iloc[rows].itertuples(), start=1)]


def load(chunks_text):
    """Join the committed index (ids and vectors) to passage text rebuilt from bnm.gov.my."""
    ids = pd.read_parquet(INDEX / "chunks.parquet")
    emb = np.load(INDEX / "embeddings.npy")
    if ids.chunk_id.tolist() != chunks_text.chunk_id.tolist():
        raise SourceChanged("rebuilt passages don't match the index")
    return Retriever(chunks_text, emb)
