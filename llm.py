"""Everything that talks to Google Gemini: embeddings for search, and writing the answer.

The API key comes from the GEMINI_API_KEY environment variable (Streamlit Cloud sets it from Secrets) or,
locally, from .streamlit/secrets.toml. Both are kept out of git.
"""
import os
import time
import tomllib
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors, types

CHAT_MODEL = "gemini-3.5-flash"  # pinned so results are reproducible; newer Flash models were often overloaded
EMBED_MODEL = "gemini-embedding-001"  # embedding-2 counts every text as a request (1,000/day free), too few to index
SECRETS = Path(__file__).parent / ".streamlit" / "secrets.toml"


class Unavailable(Exception):
    """Gemini can't be reached right now (daily free limit used up, or network trouble)."""


_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key and SECRETS.exists():
            key = tomllib.loads(SECRETS.read_text(encoding="utf-8")).get("GEMINI_API_KEY")
        if not key:
            raise Unavailable("No GEMINI_API_KEY found.")
        _client = genai.Client(api_key=key)
    return _client


def _retry(call, tries=8):
    """Wait and retry on rate limits (free tier) and temporary overload."""
    for attempt in range(tries):
        try:
            return call()
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == tries - 1:
                raise Unavailable(str(e)) from e
            time.sleep(min(60, 5 * 2 ** attempt))
        except OSError as e:  # network
            raise Unavailable(str(e)) from e


def embed(texts, task):
    """Unit-length float32 vectors; task is "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"."""
    out = []
    for i in range(0, len(texts), 40):  # small batches stay under the free 30k tokens-per-minute limit
        batch = texts[i:i + 40]
        r = _retry(lambda: client().models.embed_content(
            model=EMBED_MODEL, contents=[types.Content(parts=[types.Part(text=t)]) for t in batch],  # one vector each
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=768)))
        out += [e.values for e in r.embeddings]
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


SYSTEM = """You answer questions about Malaysia's economy for the public.
Use ONLY the numbered passages from Bank Negara Malaysia reports below, and the cpi_lookup tool for
consumer price index or inflation numbers by state, category or month.
- Cite every fact with its passage number in square brackets, like [2]. Cite cpi_lookup results as [CPI].
- Answer in the language of the question (English or Malay), in a few short sentences.
- If the passages and the tool don't contain the answer, reply exactly:
  "I couldn't find this in the reports." (in Malay: "Saya tidak menemui maklumat ini dalam laporan.")
- Never guess numbers, dates or forecasts that aren't in the passages or the tool result."""


def answer(question, passages):
    """Gemini's answer from numbered passages ({n, title, year, page, text}); may call cpi_lookup."""
    from cpi import cpi_lookup

    used = {}

    def cpi_tool(state: str = "Malaysia", category: str = "Overall", month: str | None = None) -> dict:
        result = cpi_lookup(state, category, month)
        used.update(result)
        return result
    cpi_tool.__doc__ = cpi_lookup.__doc__
    cpi_tool.__name__ = "cpi_lookup"

    context = "\n\n".join(f"[{p['n']}] {p['title']}, page {p['page']}:\n{p['text']}" for p in passages)
    r = _retry(lambda: client().models.generate_content(
        model=CHAT_MODEL, contents=f"Passages:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM, tools=[cpi_tool], temperature=0)))
    if not r.text:
        raise Unavailable("empty response")
    return {"text": r.text.strip(), "cpi": used or None}
