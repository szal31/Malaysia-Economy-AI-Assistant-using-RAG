"""Consumer Price Index lookup, used by Gemini as a tool for number questions.

Reads a frozen copy of the OpenDOSM CPI table from project 01 (Jan 2010 - Sep 2026), so answers match the
Power BI dashboard exactly.
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
_cpi = pd.read_csv(DATA / "fact_cpi.csv", dtype={"division_code": str})
_divisions = pd.read_csv(DATA / "dim_division.csv", dtype={"division_code": str})
CODE = dict(zip(_divisions.division, _divisions.division_code))
STATES = sorted(_cpi.state.unique())
LATEST = _cpi.date.max()[:7]
EARLIEST = _cpi.date.min()[:7]

ALIASES = {s.lower(): s for s in STATES} | {c.lower(): c for c in CODE} | {
    # states
    "penang": "Pulau Pinang", "kuala lumpur": "W.P. Kuala Lumpur", "kl": "W.P. Kuala Lumpur",
    "wp kuala lumpur": "W.P. Kuala Lumpur", "labuan": "W.P. Labuan", "putrajaya": "W.P. Putrajaya",
    "malacca": "Melaka", "national": "Malaysia", "nasional": "Malaysia",
    # categories, English short forms and Malay
    "all": "Overall", "headline": "Overall", "keseluruhan": "Overall",
    "food": "Food & Non-Alcoholic Beverages", "makanan": "Food & Non-Alcoholic Beverages",
    "alcohol": "Alcoholic Beverages & Tobacco", "tobacco": "Alcoholic Beverages & Tobacco",
    "clothing": "Clothing & Footwear", "pakaian": "Clothing & Footwear",
    "housing": "Housing, Water, Electricity & Gas", "perumahan": "Housing, Water, Electricity & Gas",
    "utilities": "Housing, Water, Electricity & Gas", "electricity": "Housing, Water, Electricity & Gas",
    "furnishings": "Furnishings & Household Maintenance",
    "health": "Health", "kesihatan": "Health",
    "transport": "Transport", "pengangkutan": "Transport",
    "communication": "Information & Communication", "komunikasi": "Information & Communication",
    "recreation": "Recreation, Sport & Culture", "rekreasi": "Recreation, Sport & Culture",
    "education": "Education", "pendidikan": "Education",
    "restaurants": "Restaurants & Accommodation", "restoran": "Restaurants & Accommodation",
    "insurance": "Insurance & Financial Services", "insurans": "Insurance & Financial Services",
    "personal care": "Personal Care & Misc.", "penjagaan diri": "Personal Care & Misc.",
}


def _num(x):
    return None if pd.isna(x) else float(x)


def cpi_lookup(state: str = "Malaysia", category: str = "Overall", month: str | None = None) -> dict:
    """Look up Malaysia's consumer price index and inflation from OpenDOSM.

    Args:
        state: A Malaysian state, or "Malaysia" for the national figure.
        category: A CPI category such as "Overall", "Food", "Transport" or "Housing".
        month: "YYYY-MM". Leave empty for the latest month available.

    Returns the CPI index, year-on-year inflation % (what people usually call inflation) and
    month-on-month change %.
    """
    s, c = ALIASES.get(state.strip().lower()), ALIASES.get(category.strip().lower())
    if s is None:
        return {"error": f"Unknown state '{state}'.", "valid": STATES}
    if c is None:
        return {"error": f"Unknown category '{category}'.", "valid": list(CODE)}
    if month and not re.fullmatch(r"\d{4}-\d{2}", month):
        return {"error": f"Month must look like 2024-01, got '{month}'."}
    result = {}
    if not month or month > LATEST:
        if month:
            result["note"] = f"No data for {month} yet; {LATEST} is the latest month available."
        month = LATEST
    if month < EARLIEST:
        return {"error": f"Data starts in {EARLIEST}."}
    row = _cpi[(_cpi.state == s) & (_cpi.division_code == CODE[c]) & (_cpi.date == f"{month}-01")]
    if row.empty:
        return {"error": f"No data for {s}, {c}, {month}."}
    row = row.iloc[0]
    return {"state": s, "category": c, "month": month, "cpi_index": _num(row.cpi_index),
            "inflation_yoy_pct": _num(row.inflation_yoy), "inflation_mom_pct": _num(row.inflation_mom),
            "source": "OpenDOSM CPI"} | result
