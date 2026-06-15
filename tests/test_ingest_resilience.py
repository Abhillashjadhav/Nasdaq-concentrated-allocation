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
    # _ingest now constructs one shared EdgarClient (needs a User-Agent); the
    # monkeypatched fetch_* ignore the client they receive.
    monkeypatch.setenv("STOCKSCOPE_SEC_USER_AGENT", "test test@example.com")
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


def test_ingest_shares_one_edgar_client_across_tickers(tmp_path, monkeypatch):
    """All per-ticker EDGAR calls receive the SAME client+resolver, so one throttle
    governs the whole multi-ticker run (SEC rate-limits by IP, not per ticker)."""
    from data.edgar_client import CikResolver, EdgarClient

    monkeypatch.setenv("STOCKSCOPE_SEC_USER_AGENT", "test test@example.com")
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)  # all good

    clients, resolvers = [], []

    class _Res:
        gaps: list = []

    def _cap_fund(ticker, *, client=None, resolver=None, **k):
        clients.append(client); resolvers.append(resolver); return None

    def _cap_form4(ticker, *, client=None, resolver=None, **k):
        clients.append(client); resolvers.append(resolver); return _Res()

    monkeypatch.setattr(fundamentals_mod, "fetch_fundamentals", _cap_fund)
    monkeypatch.setattr(form4_mod, "fetch_insider_buys", _cap_form4)

    store = PITStore(tmp_path / "shared.sqlite")
    _ingest(_config(store, ["AAA", "BBB", "CCC"]), store)

    # 3 tickers x 2 EDGAR sources (fundamentals + form4) = 6 calls
    assert len(clients) == 6
    assert all(isinstance(c, EdgarClient) for c in clients)
    assert len({id(c) for c in clients}) == 1            # one shared client across tickers
    assert all(isinstance(r, CikResolver) for r in resolvers)
    assert len({id(r) for r in resolvers}) == 1          # one shared resolver too


def test_ingest_form4_window_excludes_far_history(tmp_path, monkeypatch):
    """Form 4 is bounded to ~6 months before the first entry through the last entry,
    NOT the wide price window — a 2019-2021 run must not reach back to 2017."""
    import pandas as pd
    monkeypatch.setenv("STOCKSCOPE_SEC_USER_AGENT", "test test@example.com")
    monkeypatch.setattr(prices_mod, "fetch_prices", _fake_prices)
    monkeypatch.setattr(fundamentals_mod, "fetch_fundamentals", lambda *a, **k: None)

    captured = {}

    class _R:
        gaps: list = []

    def cap_form4(ticker, *, start=None, end=None, **k):
        captured["start"], captured["end"] = start, end
        return _R()

    monkeypatch.setattr(form4_mod, "fetch_insider_buys", cap_form4)

    store = PITStore(tmp_path / "f4win.sqlite")
    cfg = RunConfig(store=store, tickers=["AAA"],
                    entry_dates=[date(2019, 1, 1), date(2020, 1, 1), date(2021, 1, 1)],
                    active_signals=["momentum"], output_dir="/tmp/_f4win", ingest=True)
    _ingest(cfg, store)

    s, e = pd.Timestamp(captured["start"]), pd.Timestamp(captured["end"])
    assert s >= pd.Timestamp("2018-01-01")   # NOT 2017 (the price window's start)
    assert s < pd.Timestamp("2019-01-01")    # before the first entry (covers the insider lookback)
    assert e == pd.Timestamp("2021-01-01")   # the last entry
