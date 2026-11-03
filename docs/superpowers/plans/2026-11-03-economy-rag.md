# Malaysia Economy Assistant: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit chat app that answers questions about Malaysia's economy from Bank Negara Malaysia (BNM)
reports, with page citations, plus CPI number lookups from OpenDOSM data. It comes with a retrieval
evaluation comparing BM25, Gemini embeddings and a hybrid.

**Architecture:**
- Documents are downloaded from bnm.gov.my and split into page-bound chunks.
- An offline script embeds the chunks with Gemini and commits only vectors and page metadata, with no BNM
  text. BNM's site terms reserve copyright.
- At start-up the app downloads the same documents, checks their SHA-256 and rebuilds the identical chunk
  text.
- Retrieval uses NumPy and BM25. Gemini writes the answer from the numbered chunks and can call a
  `cpi_lookup` function.

**Tech stack:** Python 3.14 venv, `google-genai`, `rank-bm25`, `pypdf`, `beautifulsoup4`, `requests`, NumPy,
pandas, pyarrow, Streamlit, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-economy-rag-design.md`

## Global Constraints

- Venv: `py -3.14 -m venv .venv`. Never commit `.venv/`, `data/raw/`, `.streamlit/secrets.toml`, or any
  BNM PDF, HTML or chunk text.
- API key: read from `st.secrets["GEMINI_API_KEY"]` in the app and from the `GEMINI_API_KEY` env var in
  scripts. It must never be printed or logged.
- Hosted app imports must not include torch, pypdf-heavy extras, or matplotlib. App RAM must stay well under
  1 GB.
- Model ids live in one place (`llm.py`: `CHAT_MODEL`, `EMBED_MODEL`), set to the current free-tier Flash
  chat model and Gemini embedding model listed by `client.models.list()` at build time.
- CPI data: a frozen copy of project 01 `data/clean/fact_cpi.csv` (Jan 2010 – Sep 2026). "Inflation" means
  `inflation_yoy`.
- README and chart style follow `05-malay-sentiment` (results first, honest limits, the same palette
  constants as `compare.py`). Credit "Source: Bank Negara Malaysia" and OpenDOSM.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **BNM replaces or moves a PDF.** The SHA-256 no longer matches, so chunk ids no longer line up with the
   vectors. The app must say the sources changed and stop, not answer from the wrong passages
   (test in Task 1).
2. **Gemini quota or network error mid-answer.** The app shows the retrieved passages with sources and a
   "AI answer unavailable, try tomorrow" note, never a traceback (test in Task 5).
3. **CPI question for a month after the data ends, or an unknown state or category.** The tool returns the
   latest month with `note`, or an `error` naming valid options (tests in Task 3).
4. **A Malay question.** Dense or hybrid retrieval still finds English pages. The evaluation reports the
   Malay subset (Task 4).
5. **Empty or very long input.** Empty input does nothing. Input over 1,000 characters is cut with a note
   (test in Task 5).

---

### Task 1: Sources, download and chunking

**Files:**
- Create: `sources.csv`, `docs.py`, `tests/test_docs.py`, `.gitignore`, `requirements.txt`

**Interfaces:**
- `sources.csv` columns: `doc_id,title,year,kind,url,sha256`. `kind` is `pdf` or `html`. Rows:
  - The BNM Annual Report and Economic & Monetary Review for 2023, 2024 and 2025 (the 2025 editions were
    released 31 Mar 2026).
  - Every Monetary Policy Statement from Jan 2023 to the latest one (HTML pages on bnm.gov.my).

  Find the URLs on bnm.gov.my and fill `sha256` by running `python docs.py --hash`.
- `fetch(row) -> bytes`: caches in `data/raw/<doc_id>`. Raises `SourceChanged(doc_id)` if the hash differs
  from `sources.csv`.
- `pages(row, content: bytes) -> list[tuple[int, str]]`: 1-based PDF page numbers. An HTML statement is a
  single page 1, with text from the main article via BeautifulSoup.
- `chunk(pages, words=250, overlap=50) -> list[dict]` with keys `doc_id, page, chunk_no, text`. Chunks never
  cross pages. Whitespace is normalised first.
- `build_chunks() -> pandas.DataFrame` with columns `chunk_id` (`f"{doc_id}:{page}:{chunk_no}"`), `doc_id`,
  `page`, `text`. Order is deterministic.

- [ ] **Step 1: Write the failing tests**
  - `test_chunk_never_crosses_pages`: two pages of 600 and 30 words give chunks whose `page` values are only
    1 or 2. Page 2 gives exactly one chunk, and every chunk has 250 words or fewer.
  - `test_chunk_overlap`: the last 50 words of chunk 0 equal the first 50 of chunk 1 on the same page.
  - `test_fetch_rejects_changed_file`: with a cached file whose bytes don't match the row's `sha256`,
    `fetch` raises `SourceChanged`. Use `tmp_path` and monkeypatch the cache dir; no network.
- [ ] **Step 2:** `pytest tests/test_docs.py -v` → FAIL (import error)
- [ ] **Step 3:** Implement `docs.py`. Create the venv, `requirements.txt` and `.gitignore`. Fill
  `sources.csv` and its hashes.
- [ ] **Step 4:** `pytest tests/test_docs.py -v` → PASS. Then
  `python -c "import docs; d=docs.build_chunks(); print(len(d), d.doc_id.nunique())"` prints a few thousand
  chunks across every `doc_id`.
- [ ] **Step 5:** Commit.

### Task 2: Embedding index and retrieval

**Files:**
- Create: `llm.py` (embedding part), `index.py`, `retrieve.py`, `tests/test_retrieve.py`
- Output (committed): `index/chunks.parquet` (`chunk_id, doc_id, page`; **no text**),
  `index/embeddings.npy` (float32, L2-normalised, rows aligned with `chunks.parquet`)

**Interfaces:**
- `llm.client() -> genai.Client`
- `llm.embed(texts: list[str], task: str) -> np.ndarray`. `task` is `"RETRIEVAL_DOCUMENT"` or
  `"RETRIEVAL_QUERY"`. It batches and retries with backoff on HTTP 429, and returns normalised float32.
- `index.py`: runs `build_chunks()`, embeds the text, writes both files, then asserts the row counts match
  and the norms are 1 ± 1e-3.
- `retrieve.Retriever(chunks: DataFrame, emb: np.ndarray)`, where `chunks` includes `text`.
  - `.bm25(q: str, k: int) -> list[int]`
  - `.dense(qvec: np.ndarray, k: int) -> list[int]`
  - `.hybrid(q, qvec, k, rrf_k=60) -> list[int]`

  All three return row indices, best first. `.hybrid` uses reciprocal rank fusion over the top 50 of each
  method. BM25 tokenises lowercase on `\w+`.
- `retrieve.load(chunks_text: DataFrame) -> Retriever`: joins the committed index to the rebuilt text on
  `chunk_id` and asserts every id matches. A mismatch raises `SourceChanged`.

- [ ] **Step 1: Write the failing tests** (tiny fake corpus of 4 chunks with hand-made 3-d vectors)
  - `test_bm25_finds_keyword`: the query "overnight policy rate" ranks the chunk with that phrase first.
  - `test_dense_cosine`: a query vector equal to row 2 ranks row 2 first.
  - `test_hybrid_rrf`: if BM25 ranks [0,1,2] and dense ranks [1,0,3], hybrid ranks 1 and 0 above 2 and 3.
  - `test_load_detects_mismatch`: one changed `chunk_id` raises `SourceChanged`.
- [ ] **Step 2:** `pytest tests/test_retrieve.py -v` → FAIL
- [ ] **Step 3:** Implement. Run `python index.py` with `GEMINI_API_KEY` set; it prints the count and the
  file sizes.
- [ ] **Step 4:** `pytest -v` → all PASS
- [ ] **Step 5:** Commit, including the `index/` files.

### Task 3: CPI lookup tool

**Files:**
- Create: `cpi.py`, `tests/test_cpi.py`, and `data/fact_cpi.csv` plus `data/dim_division.csv` (copied from
  project 01 `data/clean/`)

**Interfaces:**
- `cpi_lookup(state: str = "Malaysia", category: str = "Overall", month: str | None = None) -> dict`
  - `month` is `"YYYY-MM"`; `None` or a month after the data ends means the latest month.
  - Returns
    `{"state", "category", "month", "cpi_index", "inflation_yoy_pct", "inflation_mom_pct", "source": "OpenDOSM CPI"}`,
    plus `"note"` when the month was substituted.
  - Unknown names return `{"error": "...", "valid": [...]}`.
- `ALIASES: dict[str, str]` maps lowercase names to canonical ones. It covers:
  - category names from `dim_division.csv` plus short forms ("food", "transport", "housing",
    "restaurants");
  - Malay names: makanan, pengangkutan, perumahan, pendidikan, kesihatan, keseluruhan;
  - "Pulau Pinang" → Penang, "WP Kuala Lumpur"/"KL" → the dataset's KL name, "Malaysia"/"national" →
    Malaysia.

  Matching ignores case.

- [ ] **Step 1: Write the failing tests** (values read from the real CSV, not hard-coded guesses):
  - `test_latest_month_default`: `cpi_lookup()["month"] == "2026-09"`, and the YoY equals the CSV row for
    Malaysia, overall, 2026-09-01.
  - `test_future_month_substituted`: `cpi_lookup(month="2026-10")` returns month `"2026-09"` and has a
    `note`.
  - `test_malay_aliases`: `cpi_lookup("Pulau Pinang", "makanan", "2024-01")` equals
    `cpi_lookup("Penang", "food", "2024-01")`.
  - `test_unknown_state`: `cpi_lookup("Atlantis")` has `error`, and `valid` contains "Sabah".
- [ ] **Step 2:** `pytest tests/test_cpi.py -v` → FAIL
- [ ] **Step 3:** Implement. Load the CSV once at module level. Note there's a function-declaration dict for
  Gemini in Task 5; keep the parameter names identical.
- [ ] **Step 4:** `pytest -v` → all PASS
- [ ] **Step 5:** Commit.

### Task 4: Retrieval evaluation

**Files:**
- Create: `eval/questions.csv`, `evaluate.py`, `tests/test_evaluate.py`
- Output: `output/retrieval.json`, `output/1_retrieval_comparison.png`

**Interfaces:**
- `eval/questions.csv` columns: `id,question,lang,type,doc_id,pages`.
  - `lang` is `en` or `ms`; `type` is `report`, `cpi` or `none`; `pages` is a `;`-separated list where any
    page counts as a hit.
  - About 30 `report` questions, at least a third of them `ms`, plus 5 `cpi` and 5 `none`.
  - Write them by reading the reports and record every page that answers each one.
- `hit_at_k(ranked: list[tuple[str,int]], gold: set[tuple[str,int]], k=5) -> float` and
  `reciprocal_rank(ranked, gold) -> float`.
- `evaluate.py retrieval`: for `type == report` only, scores BM25, dense and hybrid. It writes a table
  (all / en / ms × hit@5 / MRR) to `retrieval.json` and prints it. It draws the chart with the
  `05-malay-sentiment/compare.py` palette and highlights the chosen method. It picks the method with the
  best overall hit@5, breaking ties by MRR, and writes `"chosen"` into `retrieval.json`. The app reads it
  from there.

- [ ] **Step 1: Write the failing tests:** `hit_at_k([("a",1),("b",2)], {("b",2)}, k=1) == 0`,
  `... k=2 == 1`, `reciprocal_rank(... ) == 0.5`, and an empty gold set raises `ValueError`.
- [ ] **Step 2:** `pytest tests/test_evaluate.py -v` → FAIL
- [ ] **Step 3:** Implement. Write `questions.csv`. Run `python evaluate.py retrieval`.
- [ ] **Step 4:** `pytest -v` → PASS. Look at the chart.
- [ ] **Step 5:** Commit.

### Task 5: Answering and the Streamlit app

**Files:**
- Modify: `llm.py` (answering)
- Create: `app.py`, `.streamlit/config.toml` (light and dark themes, copied from
  `04-rapidkl-ridership/.streamlit/config.toml`), `tests/test_app.py`

**Interfaces:**
- `llm.answer(question: str, passages: list[dict]) -> dict` returns `{"text": str, "cpi": dict | None}`.
  - `passages` items: `{n, title, year, page, text}`.
  - The system prompt says:
    - answer only from the numbered passages or the `cpi_lookup` result;
    - cite as [n];
    - reply in the question's language;
    - if neither contains the answer, say exactly `"I couldn't find this in the reports."` (or the Malay
      equivalent for Malay questions).
  - It uses automatic function calling with `cpi.cpi_lookup`.
  - Quota or network errors raise `llm.Unavailable`.
- `app.py` flow:
  1. `st.cache_resource` builds `build_chunks()` and `retrieve.load()` once, with the spinner "Downloading
     Bank Negara reports (first visit only)…". `SourceChanged` shows an error and `st.stop()`.
  2. Strip the question and cut it to 1,000 characters (with a caption if cut). Empty input does nothing.
  3. Retrieve the top 5 with the `chosen` method; embedding the query may raise `Unavailable`, in which
     case fall back to BM25.
  4. Call `answer()`.
  5. Show the answer, then an expander "Sources" listing `[n] title year, page p` with the first 300
     characters of each passage, plus the CPI row if one was used.
  6. On `Unavailable`, show the sources plus `st.info("The AI answer is unavailable right now (daily free
     limit). The passages below are the most relevant pages.")`.
- Example-question buttons: 4 total, 2 English and 2 Malay, one of them a CPI question. Add a footer link to
  the GitHub repo and the Power BI dashboard.

- [ ] **Step 1: Write the failing tests** with Streamlit `AppTest`, monkeypatching `llm.embed`/`llm.answer`
  and using a tiny fake index:
  - `test_answer_and_sources_shown`: clicking an example shows the fake answer text and a "Sources" expander.
  - `test_quota_fallback`: `answer` raising `Unavailable` gives no exception, an info message, and the
    sources still shown.
  - `test_empty_input_ignored` and `test_long_input_truncated`.
- [ ] **Step 2:** `pytest tests/test_app.py -v` → FAIL
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `pytest -v` → all PASS. Run `streamlit run app.py` locally with the real key and ask
  3 example questions; check the citations point to real pages.
- [ ] **Step 5:** Commit.

### Task 6: Answer check, README and deployment

**Files:**
- Modify: `evaluate.py` (`answers` sub-command)
- Create: `README.md`, `output/answers.csv`
- Modify (project 01): `powerbi/build_report.py`, `README.md`

- [ ] **Step 1:** `python evaluate.py answers` runs every question through the app's pipeline. It writes
  `output/answers.csv` (`id, question, answer, cited`) with empty `correct` and `right_page` columns, then
  grade them by hand. `python evaluate.py summary` prints the share correct, the share citing a right page,
  and the share of `none` questions correctly refused. Its self-check fails if any grade cell is blank.
- [ ] **Step 2:** The owner spot-checks 5 graded rows before the numbers go in the README.
- [ ] **Step 3:** Write the README in project 05's style:
  - results table and chart first, then the answer-check numbers;
  - where it fails, with real examples;
  - how it works, including why the BNM text isn't in the repo;
  - "Run it", with secrets setup and no key value;
  - data credits and the CPI cut-off date.
- [ ] **Step 4:** Commit, and push once the owner gives the repo URL. The owner deploys on Streamlit Cloud
  with `GEMINI_API_KEY` in Secrets. Open the live app, ask an example, and confirm it works.
- [ ] **Step 5:** In project 01, add an action button "Ask the AI about Malaysia's economy →" to the overview
  page via `build_report.py`, using a web URL to the live app. Rerun its self-check, and link the assistant
  from the README. Then commit and push. The owner reopens and republishes in Power BI.
