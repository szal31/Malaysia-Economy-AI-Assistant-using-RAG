"""Embed every passage once with Gemini and save the search index.

Only passage ids, page numbers and vectors are saved (index/), never BNM's text.

Usage:  python index.py      (needs GEMINI_API_KEY; about 15 minutes on the free tier)

The free tier allows 1,000 embeddings a day, so finished batches are saved to data/raw/ and a stopped run
continues where it left off.
"""
import numpy as np
import pandas as pd

import llm
from reports import build_chunks
from reports import RAW
from retrieve import INDEX

PROGRESS = RAW / "embeddings_progress.parquet"


def main():
    chunks = build_chunks()
    done = pd.read_parquet(PROGRESS) if PROGRESS.exists() else pd.DataFrame(columns=["chunk_id", "vector"])
    todo = chunks[~chunks.chunk_id.isin(done.chunk_id)]
    for i in range(0, len(todo), 40):
        batch = todo.iloc[i:i + 40]
        vecs = llm.embed(batch.text.tolist(), "RETRIEVAL_DOCUMENT")
        done = pd.concat([done, pd.DataFrame({"chunk_id": batch.chunk_id, "vector": list(vecs)})])
        done.to_parquet(PROGRESS)
        print(f"{len(done)}/{len(chunks)} embedded", flush=True)
    vec = dict(zip(done.chunk_id, done.vector))
    emb = np.stack([vec[c] for c in chunks.chunk_id]).astype(np.float32)
    # Self-check: one unit vector per passage.
    assert len(emb) == len(chunks) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
    INDEX.mkdir(exist_ok=True)
    chunks[["chunk_id", "doc_id", "page"]].to_parquet(INDEX / "chunks.parquet", index=False)
    np.save(INDEX / "embeddings.npy", emb)
    print(f"{len(chunks)} passages, {emb.shape[1]}-dim vectors, "
          f"{(INDEX / 'embeddings.npy').stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
