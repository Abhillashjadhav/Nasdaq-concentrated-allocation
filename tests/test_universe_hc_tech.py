"""Universe classification eval (ARCHITECTURE.md §1, §2.2, §6, §8).

A known tech ticker -> technology; a known healthcare ticker -> healthcare; an
unrelated (bank) ticker -> excluded. The sector is sourced from SimFin's company +
industry reference data, cached in the store, and membership is point-in-time (a
name appears only once it was reporting as-of the date). Offline via an injected
reference loader — no network, no `simfin` package needed.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from universe.hc_tech import (
    HEALTHCARE,
    SECTOR_FIELD,
    TECHNOLOGY,
    classify_and_cache,
    classify_sector,
    nasdaq_hc_tech_universe,
)
from store.store import PITStore

# SimFin-shaped reference rows: Ticker -> Sector + earliest fundamentals Report Date.
REFERENCE = pd.DataFrame([
    {"Ticker": "TECHX", "Sector": "Technology", "first_report": "2015-03-31"},
    {"Ticker": "HEALX", "Sector": "Healthcare", "first_report": "2015-06-30"},
    {"Ticker": "BANKX", "Sector": "Financials", "first_report": "2015-03-31"},
    {"Ticker": "NODATE", "Sector": "Technology", "first_report": None},  # no PIT anchor
])


def _loader(frame=REFERENCE):
    """Injected reference loader: ignores api_key/cache/refresh, filters by tickers
    like the real SimFin loader does."""
    def load(*, api_key=None, cache_dir=None, refresh=False, tickers=None):
        if tickers is None:
            return frame.copy()
        wanted = {str(t).upper() for t in tickers}
        return frame[frame["Ticker"].str.upper().isin(wanted)].reset_index(drop=True)
    return load


def test_classify_sector_known_labels():
    assert classify_sector("Technology") == TECHNOLOGY
    assert classify_sector("Information Technology") == TECHNOLOGY
    assert classify_sector("Healthcare") == HEALTHCARE
    assert classify_sector("Health Care") == HEALTHCARE   # vendor spelling variant
    assert classify_sector("Financials") is None          # bank -> excluded
    assert classify_sector(None) is None
    assert classify_sector(float("nan")) is None


def test_classify_and_cache_builds_tech_healthcare_universe(tmp_path):
    store = PITStore(tmp_path / "uni.sqlite")
    symbols = ["TECHX", "HEALX", "BANKX", "NODATE", "ZZZZ"]  # ZZZZ absent from SimFin
    res = classify_and_cache(symbols, store=store, reference_loader=_loader())

    members = nasdaq_hc_tech_universe(date(2099, 1, 1), symbols, store=store)
    assert set(members) == {"TECHX", "HEALX"}    # bank excluded; no-date + unknown not classified
    assert res.n_classified == 3                 # TECHX, HEALX, BANKX all had a sector
    assert res.n_kept == 2                        # of those, only tech + healthcare are in-scope
    assert res.n_quarantined == 2                # NODATE (no report date) + ZZZZ (absent)
    assert any(g["ticker"] == "ZZZZ" for g in res.quarantine)


def test_membership_is_point_in_time(tmp_path):
    store = PITStore(tmp_path / "pit.sqlite")
    classify_and_cache(["TECHX"], store=store, reference_loader=_loader())
    # TECHX's earliest report is 2015-03-31 -> not yet "reporting" in 2014
    assert nasdaq_hc_tech_universe(date(2014, 1, 1), ["TECHX"], store=store) == []
    assert nasdaq_hc_tech_universe(date(2016, 1, 1), ["TECHX"], store=store) == ["TECHX"]


def test_classification_is_cached_not_refetched(tmp_path):
    store = PITStore(tmp_path / "cache.sqlite")
    classify_and_cache(["TECHX"], store=store, reference_loader=_loader())
    again = classify_and_cache(["TECHX"], store=store, reference_loader=_loader())
    assert again.n_cached == 1 and again.n_classified == 0  # skipped, already cached


def test_resumability_no_reload_for_cached_tickers(tmp_path):
    """After caching, a re-run does no classification work for already-cached tickers
    — even if the reference loader would error, the cache short-circuits it."""
    store = PITStore(tmp_path / "resume.sqlite")
    classify_and_cache(["TECHX", "HEALX"], store=store, reference_loader=_loader())

    res = classify_and_cache(["TECHX", "HEALX"], store=store, reference_loader=_loader())
    assert res.n_cached == 2 and res.n_classified == 0


def test_incremental_cache_persists_each_ticker(tmp_path):
    """Each in-scope ticker is persisted the moment it is computed (not batched at
    the end), so a name classified earlier is queryable immediately."""
    store = PITStore(tmp_path / "inc.sqlite")
    classify_and_cache(["TECHX", "HEALX"], store=store, reference_loader=_loader())
    assert not store.get_data(SECTOR_FIELD, "TECHX", date(2099, 1, 1)).empty
    assert not store.get_data(SECTOR_FIELD, "HEALX", date(2099, 1, 1)).empty


def test_refresh_forces_reclassification(tmp_path):
    store = PITStore(tmp_path / "refresh.sqlite")
    classify_and_cache(["TECHX"], store=store, reference_loader=_loader())
    res = classify_and_cache(["TECHX"], store=store, reference_loader=_loader(), refresh=True)
    assert res.n_classified == 1 and res.n_cached == 0  # re-classified despite the cache


def test_universe_limit_caps_processing(tmp_path):
    store = PITStore(tmp_path / "limit.sqlite")
    res = classify_and_cache(["TECHX", "HEALX", "BANKX"], store=store,
                             reference_loader=_loader(), limit=1)
    assert res.n_symbols == 1  # only the first symbol processed
    assert nasdaq_hc_tech_universe(date(2099, 1, 1), ["TECHX", "HEALX", "BANKX"], store=store) == ["TECHX"]


def test_absent_from_simfin_is_quarantined_not_cached(tmp_path):
    """A name SimFin doesn't cover is quarantined (counted) and never written, so a
    later run with better coverage retries it rather than reading a poisoned cache."""
    store = PITStore(tmp_path / "absent.sqlite")
    res = classify_and_cache(["ZZZZ"], store=store, reference_loader=_loader())
    assert res.n_quarantined == 1 and res.n_kept == 0
    assert store.get_data(SECTOR_FIELD, "ZZZZ", date(2100, 1, 1)).empty


def test_progress_logging(tmp_path, caplog):
    import logging
    store = PITStore(tmp_path / "prog.sqlite")
    with caplog.at_level(logging.INFO):
        classify_and_cache(["TECHX", "HEALX"], store=store,
                           reference_loader=_loader(), log_every=1)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("classified" in m and "kept" in m and "failed" in m for m in msgs)


def test_progress_streams_to_stdout(tmp_path, capsys):
    """Progress must reach STDOUT (not only the logging module) so it streams
    through the user's `tee` — a startup line and a per-batch progress line."""
    store = PITStore(tmp_path / "stdout.sqlite")
    classify_and_cache(["TECHX", "HEALX"], store=store,
                       reference_loader=_loader(), log_every=1)
    out = capsys.readouterr().out
    assert "building universe: 2 symbols to classify" in out  # startup, no silent gap
    assert "cached: 0, to fetch: 2" in out
    assert "classified 1/2" in out and "cache-hits" in out     # a progress line streamed
    assert "universe build complete:" in out                   # completion line
