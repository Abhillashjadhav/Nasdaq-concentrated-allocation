"""Tests for the sector backfill (``backfill_sectors``).

Guards the fix for: fundamentals ingested for many tickers, but ``classify_and_cache``
only ran over a subset, so most names have quality rows yet no ``sector`` row and the
ranker silently excludes them. The backfill re-classifies exactly the fundamentals-
without-sector set. All offline (injected reference loader, no network/simfin).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from backfill_sectors import tickers_with_fundamentals_missing_sector
from signals.quality import FIELDS
from store.store import PITStore
from universe.hc_tech import classify_and_cache, nasdaq_hc_tech_universe


def _put_fundamentals(store, ticker):
    rows = []
    for f in FIELDS:
        for ev, kd in [("2024-09-30", "2024-11-14"), ("2024-06-30", "2024-08-14")]:
            rows.append({"ticker": ticker, "field": f, "value": 100.0,
                         "event_date": pd.Timestamp(ev),
                         "knowledge_date": pd.Timestamp(kd), "source": "simfin"})
    store.put_data(pd.DataFrame(rows))


def _put_sector(store, ticker, code):
    store.put_data(pd.DataFrame([{
        "ticker": ticker, "field": "sector", "value": code,
        "event_date": pd.Timestamp("2015-01-01"),
        "knowledge_date": pd.Timestamp("2015-01-01"), "source": "simfin"}]))


def _reference_loader(frame):
    def loader(*, api_key=None, cache_dir=None, refresh=False, tickers=None):
        df = frame
        if tickers is not None:
            wanted = {str(t).upper() for t in tickers}
            df = df[df["Ticker"].astype(str).str.upper().isin(wanted)]
        return df.reset_index(drop=True)
    return loader


def test_candidate_selection_excludes_already_classified(tmp_path):
    db = tmp_path / "s.sqlite"
    store = PITStore(db)
    for t in ("AAPL", "MSFT", "LLY"):
        _put_fundamentals(store, t)
    _put_sector(store, "AAPL", 1.0)  # AAPL already classified
    # a prices-only name must NOT be picked up (no fundamentals)
    store.put_data(pd.DataFrame([{"ticker": "PRICEONLY", "field": "close", "value": 10.0,
                                  "event_date": pd.Timestamp("2024-01-01"),
                                  "knowledge_date": pd.Timestamp("2024-01-01"),
                                  "source": "prices"}]))

    cands = tickers_with_fundamentals_missing_sector(str(db))
    assert cands == ["LLY", "MSFT"]  # sorted; AAPL + PRICEONLY excluded


def test_backfill_classifies_and_expands_universe(tmp_path):
    db = tmp_path / "s.sqlite"
    store = PITStore(db)
    for t in ("MSFT", "LLY", "JPM"):
        _put_fundamentals(store, t)

    ref = pd.DataFrame([
        {"Ticker": "MSFT", "Sector": "Technology", "first_report": "2010-06-30"},
        {"Ticker": "LLY", "Sector": "Healthcare", "first_report": "2010-12-31"},
        {"Ticker": "JPM", "Sector": "Financials", "first_report": "2010-12-31"},
    ])
    cands = tickers_with_fundamentals_missing_sector(str(db))
    res = classify_and_cache(cands, store=store, reference_loader=_reference_loader(ref))

    assert res.n_classified == 3      # all three had SimFin sectors
    assert res.n_kept == 2            # MSFT + LLY kept; JPM (financials) excluded
    assert res.n_quarantined == 0
    members = nasdaq_hc_tech_universe(date(2026, 1, 1), ["MSFT", "LLY", "JPM"], store=store)
    assert sorted(members) == ["LLY", "MSFT"]


def test_no_candidates_when_all_classified(tmp_path):
    db = tmp_path / "s.sqlite"
    store = PITStore(db)
    _put_fundamentals(store, "AAPL")
    _put_sector(store, "AAPL", 1.0)
    assert tickers_with_fundamentals_missing_sector(str(db)) == []
