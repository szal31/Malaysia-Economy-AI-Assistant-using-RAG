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

CHAT_MODEL = "gemini-flash-latest"
EMBED_MODEL = "gemini-embedding-001"
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


def _retry(call, tries=6):
    """Wait and retry on rate limits (the free tier allows a limited number of requests per minute)."""
    for attempt in range(tries):
        try:
            return call()
        except errors.APIError as e:
            if e.code != 429 or attempt == tries - 1:
                raise Unavailable(str(e)) from e
            time.sleep(min(60, 5 * 2 ** attempt))
        except OSError as e:  # network
            raise Unavailable(str(e)) from e


def embed(texts, task):
    """Unit-length float32 vectors; task is "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"."""
    out = []
    for i in range(0, len(texts), 100):  # API batch limit
        batch = texts[i:i + 100]
        r = _retry(lambda: client().models.embed_content(
            model=EMBED_MODEL, contents=batch, config=types.EmbedContentConfig(task_type=task, output_dimensionality=768)))
        out += [e.values for e in r.embeddings]
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
