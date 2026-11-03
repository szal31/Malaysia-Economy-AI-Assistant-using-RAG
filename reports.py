"""Download the Bank Negara Malaysia reports and cut them into page-sized passages.

BNM's website terms reserve copyright, so the report text is never committed: the indexing script and the
app both run this to rebuild the same passages from bnm.gov.my. A SHA-256 per document (in sources.csv)
guarantees the rebuilt passages are the ones the committed embeddings were made from.

Usage:  python reports.py          (download all, print passage counts)
        python reports.py --hash   (fill the sha256 column after adding a document)
"""
import hashlib
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
SOURCES = ROOT / "sources.csv"


class SourceChanged(Exception):
    """A document on bnm.gov.my no longer matches the version the index was built from."""


def _download(row):
    try:
        r = requests.get(row["url"], timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:  # moved, removed or bnm.gov.my unreachable
        raise SourceChanged(f"{row['doc_id']}: {e}") from e
    if row["kind"] == "pdf":
        return r.content
    # HTML pages carry changing tokens, so keep (and hash) only the statement's own text
    from bs4 import BeautifulSoup
    article = BeautifulSoup(r.text, "html.parser").select_one(".journal-content-article")
    if article is None:
        raise SourceChanged(f"{row['doc_id']}: page layout changed")
    text = article.get_text(" ", strip=True).split(" See also:")[0]
    return text.encode("utf-8")


def fetch(row, check=True):
    path = RAW / row["doc_id"]
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_download(row))
    content = path.read_bytes()
    if check and hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise SourceChanged(row["doc_id"])
    return content


def pages(row, content):
    if row["kind"] == "html":
        return [(1, content.decode("utf-8"))]
    from pypdf import PdfReader
    return [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(content)).pages, start=1)]


def chunk(page_texts, words=400, overlap=60):  # 400 words keeps the index under the free 1,000 embeddings a day
    out = []
    for page, text in page_texts:
        tokens = re.sub(r"\s+", " ", text).strip().split(" ")
        if tokens == [""]:
            continue
        for no, start in enumerate(range(0, max(len(tokens) - overlap, 1), words - overlap)):
            out.append({"page": page, "chunk_no": no, "text": " ".join(tokens[start:start + words])})
    return out


def build_chunks():
    rows = []
    for row in pd.read_csv(SOURCES, dtype=str).to_dict("records"):
        for c in chunk(pages(row, fetch(row))):
            rows.append({"chunk_id": f"{row['doc_id']}:{c['page']}:{c['chunk_no']}", "doc_id": row["doc_id"],
                         "page": c["page"], "text": c["text"]})
    return pd.DataFrame(rows)


def fill_hashes():
    s = pd.read_csv(SOURCES, dtype=str)
    s["sha256"] = [hashlib.sha256(fetch(r, check=False)).hexdigest() for r in s.to_dict("records")]
    s.to_csv(SOURCES, index=False)
    print(f"hashed {len(s)} documents")


if __name__ == "__main__":
    if "--hash" in sys.argv:
        fill_hashes()
    d = build_chunks()
    print(d.groupby("doc_id").size().to_string(), f"\n{len(d)} passages from {d.doc_id.nunique()} documents")
