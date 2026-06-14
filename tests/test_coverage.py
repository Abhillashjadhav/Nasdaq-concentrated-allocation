"""Coverage / survivorship report tests (ARCHITECTURE.md §2, §8).

Proves the report (a) counts fetched-of-requested correctly, (b) names every
gap by vendor/ticker/field and quarantines it, and (c) makes a known-delisted
name detectable rather than silently absent.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from evals.coverage import coverage_report
from store.store import PITStore

DELISTED = "ENRNQ"  # a long-dead name; if present it must be detectable


def _store(tmp_path) -> PITStore:
    store = PITStore(tmp_path / "cov.sqlite")
    rows = []
    for t in ["AAPL", DELISTED]:  # MSFT deliberately absent -> a coverage gap
        rows.append({
            "ticker": t, "field": "close", "value": 50.0,
            "event_date": pd.Timestamp("2008-09-15"),
            "knowledge_date": pd.Timestamp("2008-09-15"), "source": "fixture",
        })
    store.put_data(pd.DataFrame(rows))
    return store


def test_counts_and_quarantine(tmp_path):
    store = _store(tmp_path)
    rep = coverage_report(
        store, ["close"], ["AAPL", DELISTED, "MSFT"], date(2008, 12, 31)
    )
    assert rep.n_requested == 3
    assert rep.n_fetched == 2
    assert len(rep.missing) == 1
    gap = rep.missing[0]
    assert gap["ticker"] == "MSFT" and gap["field"] == "close"
    assert gap["reason"] == "absent" and gap["vendor"] == "store"
    assert ("MSFT", "close") in rep.quarantined


def test_delisted_name_is_detectable(tmp_path):
    store = _store(tmp_path)
    rep = coverage_report(store, ["close"], ["AAPL", DELISTED], date(2008, 12, 31))
    # the dead name has data here -> present, not silently dropped
    assert ("close", DELISTED) in rep.present
    assert (DELISTED, "close") not in rep.quarantined


def test_matrix_is_human_readable(tmp_path):
    store = _store(tmp_path)
    rep = coverage_report(store, ["close"], ["AAPL", "MSFT"], date(2008, 12, 31))
    matrix = rep.matrix()
    assert "fetched 1 of 2" in matrix
    assert "✓" in matrix and "·" in matrix
    assert "MSFT/close: absent" in matrix
