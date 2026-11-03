"""Measure how well each search method finds the page that answers a question.

The right pages for each question are not hand-picked: eval/questions.csv gives a phrase (regex) from the
answer, and every report page containing it counts as a correct page.

Usage:  python evaluate.py retrieval   (writes output/retrieval.json and output/1_retrieval_comparison.png)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "output"
QUESTIONS = ROOT / "eval" / "questions.csv"
METHODS = {"bm25": "BM25 (keywords)", "dense": "Gemini embeddings", "hybrid": "Hybrid (both)"}
INK, MUTED, SURFACE, BLUE, PALE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#b9cde8"  # as in project 05


def hit_at_k(ranked, gold, k=5):
    if not gold:
        raise ValueError("question has no correct pages")
    return float(any(p in gold for p in ranked[:k]))


def reciprocal_rank(ranked, gold):
    return next((1 / (i + 1) for i, p in enumerate(ranked) if p in gold), 0.0)


def gold_pages(chunks, pattern):
    hits = chunks[chunks.text.map(lambda t: re.search(pattern, t, re.I) is not None)]
    return set(zip(hits.doc_id, hits.page))


def search(r, method, q, qvec, k):
    return {"bm25": lambda: r.bm25(q, k), "dense": lambda: r.dense(qvec, k),
            "hybrid": lambda: r.hybrid(q, qvec, k)}[method]()


def retrieval():
    import llm
    import retrieve
    from reports import build_chunks

    chunks = build_chunks()
    r = retrieve.load(chunks)
    qs = pd.read_csv(QUESTIONS).query("type == 'report'").reset_index(drop=True)
    qvecs = llm.embed(qs.question.tolist(), "RETRIEVAL_QUERY")
    rows = []
    for q, qvec in zip(qs.itertuples(), qvecs):
        gold = gold_pages(chunks, q.pattern)
        for m in METHODS:
            ranked = [(chunks.doc_id[i], chunks.page[i]) for i in search(r, m, q.question, qvec, 20)]
            rows.append({"id": q.id, "lang": q.lang, "method": m, "gold_pages": len(gold),
                         "hit5": hit_at_k(ranked, gold), "rr": reciprocal_rank(ranked, gold)})
    d = pd.DataFrame(rows)
    table = {m: {"all": _scores(d[d.method == m]), "en": _scores(d[(d.method == m) & (d.lang == "en")]),
                 "ms": _scores(d[(d.method == m) & (d.lang == "ms")])} for m in METHODS}
    chosen = max(METHODS, key=lambda m: (table[m]["all"]["hit5"], table[m]["all"]["mrr"]))
    OUT.mkdir(exist_ok=True)
    (OUT / "retrieval.json").write_text(json.dumps({"chosen": chosen, "questions": len(qs), **table}, indent=2))
    d.to_csv(OUT / "retrieval_per_question.csv", index=False)
    _chart(table, chosen)
    print(pd.DataFrame({METHODS[m]: {f"{s} {k}": v for s in ("all", "en", "ms") for k, v in table[m][s].items()}
                        for m in METHODS}).T.round(3).to_string())
    print(f"\nChosen for the app: {METHODS[chosen]} ({len(qs)} questions)")


def _scores(d):
    return {"hit5": round(d.hit5.mean(), 3), "mrr": round(d.rr.mean(), 3)}


def _chart(table, chosen):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    groups = [("all", "All questions"), ("en", "English"), ("ms", "Malay")]
    y = np.arange(len(groups))
    for j, m in enumerate(METHODS):
        vals = [table[m][g]["hit5"] for g, _ in groups]
        bars = ax.barh(y + (1 - j) * 0.27, vals, height=0.25, label=METHODS[m],
                       color=BLUE if m == chosen else PALE, edgecolor=SURFACE)
        for b, v in zip(bars, vals):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=8, color=MUTED)
            ax.text(0.01, b.get_y() + b.get_height() / 2, METHODS[m], va="center", fontsize=7,
                    color="white" if m == chosen else INK)
    ax.set_yticks(y, [g for _, g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_visible(False)
    ax.set_title("Share of questions where the right page is in the top 5 results", loc="left",
                 fontsize=11, color=INK, pad=10)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    fig.tight_layout()
    fig.savefig(OUT / "1_retrieval_comparison.png", facecolor=SURFACE)


if __name__ == "__main__":
    {"retrieval": retrieval}[sys.argv[1]]()
