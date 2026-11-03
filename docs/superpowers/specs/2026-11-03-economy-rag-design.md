# Malaysia Economy Assistant: design

**Status:** draft, waiting for the owner's approval
**Folder:** `C:\Project\06-malaysia-economy-rag` (GitHub repo name to be chosen by the owner)

## Why this exists

This is the owner's third machine-learning portfolio project. The owner is a student applying for
data/business-analyst internships and, later, machine-learning internships in Malaysia.

It pairs with project 01, the Power BI dashboard of Malaysian inflation (OpenDOSM CPI data). **The dashboard
shows what happened; this assistant explains why.** A recruiter clicks a button on the dashboard, lands in a
chat, and asks things like "Why did inflation rise in 2022?" or "What was food inflation in Sabah last month?".

What should impress an ML interviewer:
1. **Every answer shows its sources** (report name and page number), so it can be checked.
2. **It is measured, not just demoed.** A small test set shows how often retrieval finds the right page,
   comparing three retrieval methods, as the earlier projects compared models against baselines.
3. **It admits when it doesn't know** instead of making things up.

## Goals

- A chat web app that answers questions about Malaysia's economy from Bank Negara Malaysia (BNM) reports,
  with citations to report and page.
- Answers CPI number questions (by state, category and month) from the same OpenDOSM data the dashboard uses,
  so the numbers match the dashboard exactly.
- Understands questions in Malay as well as English. It answers in the language of the question.
- Says "I couldn't find this in the reports" when the sources don't contain the answer.
- A retrieval evaluation with a results table and chart in the README, in the style of projects 02 and 05.
- Hosted free on Streamlit Community Cloud; the Power BI dashboard links to it.

## Non-goals

- Multi-turn memory: each question is answered on its own. (Follow-ups like "and in 2023?" may not work;
  the README says so.)
- Uploading your own documents, user accounts, or saving chat history.
- A vector database or LangChain-style framework. The index is small (a few thousand chunks) and fits in
  NumPy arrays; plain code is easier to explain in an interview.
- Forecasting or financial advice.

## Anti-goals (would count as failure even if it works)

- An answer that states a number or fact with no source, or with a source that doesn't say it.
- The Gemini API key appearing anywhere in the repo, logs or screenshots.
- A demo that breaks on the first visit because of memory limits or a slow start.
- Copyrighted report PDFs committed to GitHub without checking BNM's terms.

## Constraints

- **Hardware:** the owner's laptop is CPU-only with 12 GB RAM, often with little free. No local language model.
- **Hosting:** Streamlit Community Cloud free tier, about 1 GB RAM. Keep the app light: no PyTorch.
- **Language model:** Google Gemini through the free API key from Google AI Studio (decided with the owner).
  The key is read from Streamlit secrets (`GEMINI_API_KEY`); locally from `.streamlit/secrets.toml`, which
  is gitignored. The free tier has daily limits; the app must show a friendly message when they're hit.
- **Python:** a 3.14 virtual environment (`py -3.14 -m venv .venv`), as in projects 02 and 05. The main
  Python 3.15 can't install some packages.
- **Owner's conventions:** README style as in projects 02/05 (results first, honest limits, "Run it"
  section). Charts in the same palette as `05-malay-sentiment/compare.py`. Commits end with the Claude
  co-author line. Never commit `.venv`, secrets or large raw files.

## How it works (decided)

1. **Documents.** BNM's yearly Economic and Monetary Review / Annual Report for the last 3–4 years, plus the
   Monetary Policy Statements from the same period. English PDFs from bnm.gov.my. A download script fetches
   them; the PDFs themselves are not committed unless BNM's terms allow it.
2. **Chunking.** Extract text page by page, split into chunks of a few hundred words that never cross a page,
   and keep the report name, year and page number with each chunk.
3. **Retrieval:** three methods compared in the evaluation.
   - BM25 keyword search.
   - Gemini embeddings (multilingual, so Malay questions can match English text).
   - Hybrid: both, merged by reciprocal rank fusion.

   The best on the evaluation is used in the app. Embeddings are computed once and saved as files in the
   repo, so the app only embeds the question.
4. **Answering.** Gemini gets the question and the top retrieved chunks, numbered. It must answer only from
   them, cite them as [1], [2], and say it couldn't find the answer if they don't contain it.
5. **CPI numbers.** Gemini can call one function, `cpi_lookup(state, category, month)`, which reads a snapshot
   of project 01's `data/clean/fact_cpi.csv` (Jan 2010 – Sep 2026, 16 states plus Malaysia, 13 categories plus
   Overall). Its answer is cited as "OpenDOSM CPI, <month>".
6. **App.** A Streamlit chat page. Each answer has a "Sources" section listing report, year and page, with the
   quoted passage. A few example questions are shown as buttons, in English and Malay. Light and dark themes
   as in the Rapid KL app.
7. **Dashboard link.** A button on the Power BI overview page, "Ask the AI about Malaysia's economy", opening
   the app. It is added via project 01's `powerbi/build_report.py`. The owner republishes in Power BI.

## Evaluation (decided in outline)

- **About 30 questions with known answers**, each tagged with the report and page that answer it, written by
  reading the reports. A mix of English and Malay, and some CPI number questions.
- **5 or more questions the reports can't answer**, to check that the assistant says so.
- **Retrieval metrics:** for BM25, embeddings and hybrid, the share of questions where the right page is in
  the top 5 (hit rate @5) and how high it ranks (MRR). Results go into a table and a chart.
- **Answer check:** for each test question, record whether the answer is correct and cites the right page,
  and whether the unanswerable questions are refused. Grading by hand is fine at this size. The README reports
  the numbers honestly, including failures.

## Calls made by the designer (not discussed with the owner; owner to check)

- **CPI tool returns all three numbers** for the matched row: index, year-on-year % and month-on-month %.
  "Inflation" means year-on-year, as on the dashboard.
- **"Last month" or no month given → the latest month in the data,** and the answer names that month
  ("September 2026 is the latest available"). It never pretends a missing month exists.
- **Names are matched with a small alias table:** category names and common short forms from
  `dim_division.csv`, plus Malay names ("makanan", "pengangkutan", "Pulau Pinang" → Penang, "Malaysia" for the
  national figure). An unknown name returns an error that Gemini passes on politely.
- **The CPI data is a frozen snapshot** copied into the repo (Jan 2010 – Sep 2026). The README names the
  cut-off date; refreshing it means re-copying the file from project 01.
- **BNM terms (checked 3 Nov 2026):** BNM's website terms reserve copyright, so the repo holds only
  embeddings and page numbers; the app downloads the documents itself. The original rule was: check them at build time. If reproduction with acknowledgement is allowed, commit the
  extracted chunk text (not the PDFs). If not, commit only the embeddings and page numbers, and have the
  app download the PDFs from bnm.gov.my on first start and cache them. Either way the README credits BNM.
- **One shared free Gemini key** (the owner's) serves the demo. Document embeddings are computed once, offline.
  When the daily limit is hit, the app still shows the retrieved passages with their sources and says the AI
  answer is unavailable until tomorrow. It never shows a raw error.
- **Choosing the retrieval method:** on the whole test set, of which at least a third of the questions are in
  Malay. The README also reports the Malay subset separately, which shows why BM25 alone isn't enough.
- **Test questions:** written by the builder from reading the reports. Each lists every page that answers it,
  and any of them counts as a hit. CPI number questions are left out of the retrieval metrics and checked
  only in the answer check. The owner spot-checks a few before the results are published.
- **PDF reading** (`pypdf` or similar) is used only by the offline preparation script. The hosted app needs only
  Streamlit, the Gemini SDK, NumPy, pandas and a BM25 library, so it stays far below the 1 GB limit.

## Left to the builder

- Exact report editions and years, and the download URLs.
- Chunk size and overlap; how many chunks go to Gemini.
- Which Gemini chat and embedding models: the current free-tier models at build time.
- File layout and script names, following projects 02/05 (`prepare.py`, `evaluate.py`, `app.py`, ...).
- Wording of the prompt, the example questions and the README.
- The repo name, suggested to the owner (e.g. *Malaysia Economy AI Assistant Using RAG*), matching their
  existing naming style.
