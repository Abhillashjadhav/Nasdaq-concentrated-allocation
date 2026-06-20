"""Ingest resilience tests (ARCHITECTURE.md §2, §5).

A per-ticker fetch failure (a delisted name with no free data, or a name SimFin
doesn't cover) must be caught, quarantined with a reason, and skipped — for both
prices AND fundamentals — letting the rest of the universe proceed. The run fails
only if NO universe ticker ingests a price. Offline: prices is monkeypatched and
the SimFin bulk loader is injected (no network).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import data.prices as prices_mod
import data.simfin_client as simfin_mod
from data.prices import DataPullError
from run import PipelineError, RunConfig, _ingest
from store.schema import COLUMNS
from store.store import PITStore

BAD = "BADTICKER"


def _good_records(ticker):
    d = pd.Timestamp("2019-06-03")
    return pd.DataFrame([{"ticker": ticker, "field": "close", "value": 100.0,
                          "event_date": d, "knowledge_date": d, "source": "fake"}],
                        columns=COLUMNS)


def _fake_prices(ticker, start, end, **kwargs):
    if ticker == BAD:
        raise DataPullError(f"no price data for {ticker!r}", quarantine={"yfinance": "x"})
    return _good_records(ticker)


def _simfin_frames_for(*present, **kwargs):
    """SimFin bulk frames (blank Publish Date, free-tier shape) covering only the
    ``present`` tickers — any other requested ticker is absent and gets quarantined."""
    def inc(t):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": "2018-12-31",
                "Publish Date": "", "Revenue": 34100e6, "Gross Profit": 26900e6,
                "Net Income": 5240e6, "Shares (Diluted)": 950e6}

    def bal(t):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": "2018-12-31",
                "Publish Date": "", "Total Assets": 64000e6, "Total Current Assets": 27000e6,
                "Total Current Liabilities": 26000e6, "Long Term Debt": 18000e6}

    def cf(t):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": "2018-12-31",
                "Publish Date": "", "Net Cash from Operating Activities": 4240e6}

    return {"income": pd.DataFrame([inc(t) for t in present]),
            "balance": pd.DataFrame([bal(t) for t in present]),
            "cashflow": pd.DataFrame([cf(t) for t in present])}


def _config(store, tickers):
    return RunConfig(store=store, tickers=tickers, entry_dates=[date(2019, 1, 1)],
                     active_signals=["momentum"], output_dir="/tmp/_ingest_test", ingest=True)


def test_per_ticker_failures_quarantined_good_survives(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "ing.sqlite")
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    # SimFin covers GOOD but NOT the bad ticker.
    monkeypatch.setattr(simfin_mod, "_download_frames",
                        lambda **k: _simfin_frames_for("GOOD"))

    quarantine = _ingest(_config(store, ["GOOD", BAD]), store)

    # good ticker (and benchmark) ingested; bad ticker did not
    assert not store.get_data("close", "GOOD", date(2020, 1, 1)).empty
    assert not store.get_data("close", "^IXIC", date(2020, 1, 1)).empty
    assert store.get_data("close", BAD, date(2020, 1, 1)).empty

    by_ticker_field = {(g["ticker"], g["field"]): g for g in quarantine}
    # the bad ticker is quarantined for BOTH sources, each with a reason
    assert (BAD, "close") in by_ticker_field
    assert (BAD, "fundamentals") in by_ticker_field
    assert by_ticker_field[(BAD, "close")]["reason"].startswith("price_unavailable")
    assert by_ticker_field[(BAD, "fundamentals")]["reason"] == "absent_from_simfin"
    # the good ticker is never quarantined — a per-ticker failure was not fatal
    assert not any(g["ticker"] == "GOOD" for g in quarantine)


def _blank_publish_simfin_frames(**kwargs):
    """Real free-tier shape: SimFin bulk frames with Publish Date as an EMPTY STRING,
    two annual periods per name so quality has enough history to score."""
    def inc(t, rep):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": rep, "Publish Date": "",
                "Revenue": 34100e6, "Gross Profit": 26900e6, "Net Income": 5240e6,
                "Shares (Diluted)": 950e6}

    def bal(t, rep):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": rep, "Publish Date": "",
                "Total Assets": 64000e6, "Total Current Assets": 27000e6,
                "Total Current Liabilities": 26000e6, "Long Term Debt": 18000e6}

    def cf(t, rep):
        return {"Ticker": t, "Fiscal Period": "FY", "Report Date": rep, "Publish Date": "",
                "Net Cash from Operating Activities": 4240e6}

    i, b, c = [], [], []
    for t in ("LLY", "AAPL"):
        for rep in ("2022-12-31", "2023-12-31"):
            i.append(inc(t, rep)); b.append(bal(t, rep)); c.append(cf(t, rep))
    return {"income": pd.DataFrame(i), "balance": pd.DataFrame(b), "cashflow": pd.DataFrame(c)}


def test_ingest_simfin_blank_publish_populates_quality_end_to_end(tmp_path, monkeypatch):
    """END-TO-END real-path guard: the run.py ``_ingest`` path (NOT the diagnostic, NOT
    records_from_frames in isolation) must persist SimFin fundamentals into the store
    even when Publish Date ships blank, so quality reads them and is not n/a. This is
    the gap that let "diagnostic OK but ranking n/a" be possible — now covered."""
    from signals.quality import quality_score

    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    monkeypatch.setattr(simfin_mod, "_download_frames",
                        lambda **k: _blank_publish_simfin_frames())

    store = PITStore(tmp_path / "e2e.sqlite")
    cfg = RunConfig(store=store, tickers=["LLY", "AAPL"], entry_dates=[date(2024, 6, 30)],
                    active_signals=["quality"], output_dir=str(tmp_path), ingest=True)
    _ingest(cfg, store)  # the exact path run.py drives on --ingest

    for tkr in ("LLY", "AAPL"):
        q = quality_score(tkr, date(2024, 6, 30), store=store)
        assert not q.insufficient_data, f"{tkr}: {q.reason}"  # populated from the real store
        assert q.score is not None


def test_zero_priced_tickers_is_fatal(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "ing2.sqlite")
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    monkeypatch.setattr(simfin_mod, "_download_frames", lambda **k: _simfin_frames_for())
    with pytest.raises(PipelineError):
        _ingest(_config(store, [BAD]), store)  # the only ticker's price fetch fails


def test_simfin_loader_receives_universe_tickers(tmp_path, monkeypatch):
    """The ingest path must hand the universe ticker list to the SimFin loader so it
    filters the bulk CSVs early (the perf guard) instead of processing all companies."""
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return _simfin_frames_for("GOOD")

    monkeypatch.setattr(simfin_mod, "_download_frames", _spy)
    store = PITStore(tmp_path / "spy.sqlite")
    _ingest(_config(store, ["GOOD"]), store)
    assert seen.get("tickers") == ["GOOD"]
