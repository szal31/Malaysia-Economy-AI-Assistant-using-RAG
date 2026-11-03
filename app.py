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
REPO = "https://github.com/szal31/Malaysia-Economy-AI-Assistant-using-RAG"
DASHBOARD = ("https://app.powerbi.com/view?r=eyJrIjoiZmZjNDg2ODItNzJhMy00OTllLWJmNTYtM2Q5MDg4YWFiMGQxIiwidCI6ImN"
             "kY2JiMGUyLTlmZWEtNGY1NC04NjcwLTY3MjcwNzc5N2FkYSIsImMiOjEwfQ%3D%3D")
MAX_CHARS, TOP_K = 1000, 5
EXAMPLES = ["Why did inflation rise in 2022?", "Bilakah BNM menurunkan OPR kepada 2.75%?",
            "What was food inflation in Sabah last month?", "Unjuran pertumbuhan ekonomi 2026?"]

st.set_page_config(page_title="Malaysia Economy Assistant", page_icon=":material/account_balance:")


@st.cache_resource(show_spinner="Downloading Bank Negara reports (first visit only, about a minute)...")
def retriever():
    return retrieve.load(reports.build_chunks())


@st.cache_data
def sources():
    return pd.read_csv(ROOT / "sources.csv", dtype=str).set_index("doc_id")


def titles():
    return sources().title.to_dict()


def link(p):
    url = sources().url[p["doc_id"]]
    return f"{url}#page={p['page']}" if url.endswith(".pdf") else url  # PDFs open at the cited page


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
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(turn["question"])
        if turn["cut"]:
            st.caption(f"Your question was shortened to the first {MAX_CHARS:,} characters.")
    with st.chat_message("assistant", avatar=":material/account_balance:"):
        if turn["result"] is None:
            st.info("The AI answer is unavailable right now (the free daily limit may be used up). "
                    "The report pages below are the closest matches to your question.", icon=":material/cloud_off:")
        else:
            st.markdown(turn["result"]["text"])
        with st.expander(f"Sources · {len(turn['passages'])} report pages", icon=":material/menu_book:"):
            c = turn["result"] and turn["result"]["cpi"]
            if c and "error" not in c:
                st.markdown(f"**[CPI]** OpenDOSM consumer price index: {c['state']}, {c['category']}, "
                            f"{c['month']}: **{c['inflation_yoy_pct']}% year on year**")
                if c.get("note"):
                    st.caption(c["note"])
            for p in turn["passages"]:
                st.markdown(f"**[{p['n']}]** [{p['title']}, page {p['page']}]({link(p)})")
                st.caption(p["text"][:220].replace("$", r"\$") + "…")  # $ would start LaTeX


st.caption(":material/account_balance: Retrieval-augmented answers from Bank Negara Malaysia reports")
st.title("Malaysia Economy Assistant")
st.markdown("Ask about growth, inflation and interest rates **in English or Bahasa Melayu**. "
            "Every answer cites the report page it came from.")
st.markdown(":blue-badge[:material/description: 33 BNM reports and statements, 2022–2026] "
            ":blue-badge[:material/query_stats: State inflation to Sep 2026] "
            ":green-badge[:material/check_circle: 90% of test answers correct]")
st.caption(f"Each question is answered on its own, so follow-ups like \"and in 2023?\" need the full question. "
           f"[Inflation dashboard ↗]({DASHBOARD}) · [How it works ↗]({REPO})")

if "history" not in st.session_state:
    st.session_state.history = []
# an example click is stored by its callback before this rerun, so the starter list is already gone
question = st.chat_input("Ask about inflation, GDP or the OPR, in English or Bahasa Melayu") or \
    st.session_state.pop("example", None)
if not st.session_state.history and not question:  # starter questions only on an empty chat
    st.markdown("**Try a question**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        cols[i % 2].button(q, key=f"ex{i}", width="stretch", on_click=st.session_state.update, kwargs={"example": q})

for turn in st.session_state.history:
    show(turn)
if question and question.strip():
    try:
        with st.spinner("Reading the reports..."):
            turn = ask(question.strip())
    except reports.SourceChanged:
        st.error("A Bank Negara source document has changed or bnm.gov.my can't be reached right now, so the "
                 "reports can't be loaded. Please try again later.")
        st.stop()
    st.session_state.history.append(turn)
    show(turn)
