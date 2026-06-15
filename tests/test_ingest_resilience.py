"""Ingest resilience tests (ARCHITECTURE.md §2, §5).

A per-ticker fetch failure (a delisted name with no free data, or a transient
provider error) must be caught, quarantined with a reason, and skipped — for
prices, fundamentals, AND Form 4 — letting the rest of the universe proceed. The
run fails only if NO universe ticker ingests a price. Offline: all three adapters
are monkeypatched (no network).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import data.form4 as form4_mod
import data.fundamentals as fundamentals_mod
import data.prices as prices_mod
from data.edgar_client import UnknownTickerError
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


def _fake_edgar(ticker, *args, **kwargs):  # fundamentals / form4 stand-in
    if ticker == BAD:
        raise UnknownTickerError(f"{ticker!r} not found")
    return None


def _patch_all(monkeypatch):
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    monkeypatch.setattr(fundamentals_mod, "fetch_fundamentals", _fake_edgar)
    monkeypatch.setattr(form4_mod, "fetch_insider_buys", _fake_edgar)


def _config(store, tickers):
    return RunConfig(store=store, tickers=tickers, entry_dates=[date(2019, 1, 1)],
                     active_signals=["momentum"], output_dir="/tmp/_ingest_test", ingest=True)


def test_per_ticker_failures_quarantined_good_survives(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "ing.sqlite")
    _patch_all(monkeypatch)

    quarantine = _ingest(_config(store, ["GOOD", BAD]), store)

    # good ticker (and benchmark) ingested; bad ticker did not
    assert not store.get_data("close", "GOOD", date(2020, 1, 1)).empty
    assert not store.get_data("close", "^IXIC", date(2020, 1, 1)).empty
    assert store.get_data("close", BAD, date(2020, 1, 1)).empty

    by_ticker_field = {(g["ticker"], g["field"]): g for g in quarantine}
    # the bad ticker is quarantined for ALL THREE sources, each with a reason
    assert (BAD, "close") in by_ticker_field
    assert (BAD, "fundamentals") in by_ticker_field
    assert (BAD, "form4_buy_P") in by_ticker_field
    assert by_ticker_field[(BAD, "close")]["reason"].startswith("price_unavailable")
    assert by_ticker_field[(BAD, "fundamentals")]["reason"].startswith("fundamentals_unavailable")
    assert by_ticker_field[(BAD, "form4_buy_P")]["reason"].startswith("form4_unavailable")
    # the good ticker is never quarantined — a per-ticker failure was not fatal
    assert not any(g["ticker"] == "GOOD" for g in quarantine)


def test_zero_priced_tickers_is_fatal(tmp_path, monkeypatch):
    store = PITStore(tmp_path / "ing2.sqlite")
    _patch_all(monkeypatch)
    with pytest.raises(PipelineError):
        _ingest(_config(store, [BAD]), store)  # the only ticker's price fetch fails
