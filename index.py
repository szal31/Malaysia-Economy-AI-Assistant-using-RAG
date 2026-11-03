"""Embed every passage once with Gemini and save the search index.

Only passage ids, page numbers and vectors are saved (index/), never BNM's text.

Usage:  python index.py      (needs GEMINI_API_KEY; takes a few minutes on the free tier)
"""
import numpy as np

import llm
from reports import build_chunks
from retrieve import INDEX


def main():
    chunks = build_chunks()
    emb = llm.embed(chunks.text.tolist(), "RETRIEVAL_DOCUMENT")
    # Self-check: one unit vector per passage.
    assert len(emb) == len(chunks) and np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-3)
    INDEX.mkdir(exist_ok=True)
    chunks[["chunk_id", "doc_id", "page"]].to_parquet(INDEX / "chunks.parquet", index=False)
    np.save(INDEX / "embeddings.npy", emb)
    print(f"{len(chunks)} passages, {emb.shape[1]}-dim vectors, "
          f"{(INDEX / 'embeddings.npy').stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
