import numpy as np
import pandas as pd
import pytest

import retrieve
from reports import SourceChanged

TEXTS = ["the overnight policy rate was kept at 2.75%", "exports grew strongly", "food prices rose",
         "the ringgit appreciated"]
EMB = np.eye(4, 3, dtype=np.float32)
EMB[3] = [0.6, 0.8, 0]


def corpus():
    return pd.DataFrame({"chunk_id": [f"d:{i}:0" for i in range(4)], "doc_id": "d", "page": range(4), "text": TEXTS})


def test_bm25_finds_keyword():
    assert retrieve.Retriever(corpus(), EMB).bm25("overnight policy rate", 2)[0] == 0


def test_dense_cosine():
    assert retrieve.Retriever(corpus(), EMB).dense(EMB[2], 2)[0] == 2


def test_hybrid_rrf(monkeypatch):
    r = retrieve.Retriever(corpus(), EMB)
    monkeypatch.setattr(r, "bm25", lambda q, k: [0, 1, 2][:k])
    monkeypatch.setattr(r, "dense", lambda v, k: [1, 0, 3][:k])
    assert set(r.hybrid("q", EMB[0], 4)[:2]) == {0, 1}


def test_load_detects_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "INDEX", tmp_path)
    corpus()[["chunk_id", "doc_id", "page"]].to_parquet(tmp_path / "chunks.parquet")
    np.save(tmp_path / "embeddings.npy", EMB)
    assert len(retrieve.load(corpus()).chunks) == 4
    changed = corpus()
    changed.loc[3, "chunk_id"] = "d:9:0"
    with pytest.raises(SourceChanged):
        retrieve.load(changed)
