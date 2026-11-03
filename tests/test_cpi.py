import pandas as pd

import cpi


def test_latest_month_default():
    r = cpi.cpi_lookup()
    d = pd.read_csv("data/fact_cpi.csv", dtype={"division_code": str})
    row = d[(d.state == "Malaysia") & (d.division_code == "overall") & (d.date == "2026-09-01")].iloc[0]
    assert r["month"] == "2026-09" and "note" not in r
    assert r["inflation_yoy_pct"] == row.inflation_yoy and r["cpi_index"] == row.cpi_index


def test_future_month_substituted():
    r = cpi.cpi_lookup(month="2026-10")
    assert r["month"] == "2026-09" and "note" in r


def test_malay_aliases():
    assert cpi.cpi_lookup("Pulau Pinang", "makanan", "2024-01") == cpi.cpi_lookup("Penang", "food", "2024-01")
    assert cpi.cpi_lookup("KL", "Overall")["state"] == "W.P. Kuala Lumpur"


def test_unknown_state():
    r = cpi.cpi_lookup("Atlantis")
    assert "error" in r and "Sabah" in r["valid"]


def test_month_before_data_is_error():
    assert "error" in cpi.cpi_lookup(month="1999-01")


def test_bad_month_format():
    assert "error" in cpi.cpi_lookup(month="Sept 2026")
