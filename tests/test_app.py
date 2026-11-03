import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import llm
import reports
import retrieve

CHUNKS = pd.DataFrame({"chunk_id": ["emr2024:38:0", "mps20250709:1:0"], "doc_id": ["emr2024", "mps20250709"],
                       "page": [38, 1], "text": ["Headline inflation moderated in 2024, averaging at 1.8%.",
                                                 "The MPC decided to reduce the OPR by 25 basis points to 2.75%."]})


@pytest.fixture
def app(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(reports, "build_chunks", lambda: CHUNKS)
    monkeypatch.setattr(retrieve, "load", lambda c: retrieve.Retriever(c, np.eye(2, dtype=np.float32)))
    monkeypatch.setattr(llm, "embed", lambda texts, task: np.eye(2, dtype=np.float32)[:len(texts)])
    monkeypatch.setattr(llm, "answer", lambda q, p: {"text": "Inflation averaged 1.8% [1].", "cpi": None})
    return AppTest.from_file("../app.py", default_timeout=30)


def ask(at, text):
    at.run()
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def test_answer_and_sources_shown(app):
    at = ask(app, "What was inflation in 2024?")
    assert "Inflation averaged 1.8% [1]." in [m.value for m in at.markdown]
    assert any(e.label.startswith("Sources") for e in at.expander)


def test_quota_fallback(app, monkeypatch):
    def broken(q, p):
        raise llm.Unavailable("429")
    monkeypatch.setattr(llm, "answer", broken)
    at = ask(app, "What was inflation in 2024?")
    assert at.info and "unavailable" in at.info[0].value
    assert any(e.label.startswith("Sources") for e in at.expander)


def test_example_button_asks(app):
    app.run()
    app.button[0].click().run()
    assert not app.exception and any(e.label.startswith("Sources") for e in app.expander)


def test_long_input_truncated(app):
    at = ask(app, "inflation " * 300)
    assert any("1,000 characters" in c.value for c in at.caption)


def test_blank_input_ignored(app):
    at = ask(app, "   ")
    assert not at.expander


def test_sources_link_to_the_report_page(app):
    at = ask(app, "What was inflation in 2024?")
    links = " ".join(m.value for m in at.markdown)
    assert "emr2024_en_book.pdf#page=38" in links  # PDFs open at the cited page
    assert "monetary-policy-statement-09072025" in links
