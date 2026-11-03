"""Chat with Bank Negara Malaysia's reports about Malaysia's economy, with page citations.

Usage:  streamlit run app.py      (needs GEMINI_API_KEY in .streamlit/secrets.toml)
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import llm
import reports
import retrieve

ROOT = Path(__file__).parent
REPO = "https://github.com/szal31/Malaysia-Economy-AI-Assistant-Using-RAG"
DASHBOARD = ("https://app.powerbi.com/view?r=eyJrIjoiZmZjNDg2ODItNzJhMy00OTllLWJmNTYtM2Q5MDg4YWFiMGQxIiwidCI6ImN"
             "kY2JiMGUyLTlmZWEtNGY1NC04NjcwLTY3MjcwNzc5N2FkYSIsImMiOjEwfQ%3D%3D")
MAX_CHARS, TOP_K = 1000, 5
EXAMPLES = ["Why did inflation rise in 2022?", "Bilakah BNM menurunkan OPR kepada 2.75%?",
            "What was food inflation in Sabah last month?", "Berapakah unjuran pertumbuhan ekonomi bagi 2026?"]

st.set_page_config(page_title="Malaysia Economy Assistant", page_icon=":material/forum:")


@st.cache_resource(show_spinner="Downloading Bank Negara reports (first visit only, about a minute)...")
def retriever():
    return retrieve.load(reports.build_chunks())


@st.cache_data
def titles():
    return pd.read_csv(ROOT / "sources.csv", dtype=str).set_index("doc_id").title.to_dict()


def method():
    path = ROOT / "output" / "retrieval.json"
    return json.loads(path.read_text())["chosen"] if path.exists() else "hybrid"


def ask(question):
    cut = len(question) > MAX_CHARS
    question = question[:MAX_CHARS]
    passages = retrieve.passages(retriever(), question, method(), titles(), TOP_K)
    try:
        result = llm.answer(question, passages)
    except llm.Unavailable:
        result = None
    return {"question": question, "cut": cut, "result": result, "passages": passages}


def show(turn):
    with st.chat_message("user"):
        st.markdown(turn["question"])
        if turn["cut"]:
            st.caption(f"Your question was shortened to the first {MAX_CHARS:,} characters.")
    with st.chat_message("assistant"):
        if turn["result"] is None:
            st.info("The AI answer is unavailable right now (the free daily limit may be used up). "
                    "The passages below are the most relevant report pages.")
        else:
            st.markdown(turn["result"]["text"])
        with st.expander(f"Sources ({len(turn['passages'])} report pages)"):
            if turn["result"] and turn["result"]["cpi"]:
                c = turn["result"]["cpi"]
                st.markdown(f"**[CPI] OpenDOSM consumer price index:** {c}")
            for p in turn["passages"]:
                st.markdown(f"**[{p['n']}] {p['title']}, page {p['page']}**  \n{p['text'][:300]}...")


st.title("Malaysia Economy Assistant")
st.markdown("Ask about Malaysia's economy, inflation and interest rates, in English or Malay. Answers come from "
            "Bank Negara Malaysia's Economic and Monetary Reviews (2022–2025) and Monetary Policy Statements, "
            "and OpenDOSM price data up to September 2026, with the page for every fact.")
st.caption(f"Each question is answered on its own; follow-ups like \"and in 2023?\" won't remember the last one. "
           f"[See the inflation dashboard]({DASHBOARD}) · [How it works]({REPO})")

if "history" not in st.session_state:
    st.session_state.history = []
cols = st.columns(2)
clicked = next((q for i, q in enumerate(EXAMPLES) if cols[i % 2].button(q, key=f"ex{i}", width="stretch")), None)
question = st.chat_input("Ask a question") or clicked

for turn in st.session_state.history:
    show(turn)
if question and question.strip():
    try:
        with st.spinner("Reading the reports..."):
            turn = ask(question.strip())
    except reports.SourceChanged:
        st.error("Bank Negara has changed one of the source documents, so the search index needs rebuilding. "
                 "Please try again later.")
        st.stop()
    st.session_state.history.append(turn)
    show(turn)
