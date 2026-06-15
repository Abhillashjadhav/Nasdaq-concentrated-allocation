"""Benchmark / index-symbol handling for the price adapter.

The Nasdaq Composite benchmark is ^IXIC (the yfinance symbol). Stooq codes
indices differently (^ndq, not the .us equity form), so the adapter must map the
symbol per provider. These tests are offline (Stooq via a captured fake request).
"""

from __future__ import annotations

from datetime import date

import pytest

from backtest.labels import BENCHMARK_TICKER
from data import prices
from data.prices import _stooq_symbol, fetch_prices


def test_benchmark_is_ixic():
    assert BENCHMARK_TICKER == "^IXIC"


def test_stooq_symbol_index_vs_equity():
    assert _stooq_symbol("^IXIC") == "^ndq"      # Nasdaq Composite index code
    assert _stooq_symbol("^GSPC") == "^spx"
    assert _stooq_symbol("AAPL") == "aapl.us"    # US equity keeps the .us suffix
    assert _stooq_symbol("^OTHER") == "^other"   # generic index, best-effort


class _Resp:
    text = "Date,Open,High,Low,Close,Volume\n2020-01-02,1,1,1,9000,0\n"

    def raise_for_status(self):
        pass


def test_index_benchmark_uses_stooq_index_code(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return _Resp()

    # yfinance isn't installed here, so the fetch falls through to Stooq.
    monkeypatch.setattr(prices.requests, "get", fake_get)
    recs = fetch_prices("^IXIC", date(2020, 1, 1), date(2020, 1, 3),
                        retries=1, sleep=lambda *_: None)

    assert "s=^ndq" in captured["url"]            # mapped to Stooq's index code
    assert ".us" not in captured["url"]           # not treated as a US equity
    assert recs.iloc[0]["ticker"] == "^IXIC"      # stored under the canonical symbol
    assert recs.iloc[0]["value"] == 9000.0
    assert recs.iloc[0]["source"] == "stooq"


def test_equity_still_uses_us_suffix(monkeypatch):
    captured = {}
    monkeypatch.setattr(prices.requests, "get",
                        lambda url, timeout=None: captured.update(url=url) or _Resp())
    fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 3),
                 retries=1, sleep=lambda *_: None)
    assert "s=aapl.us" in captured["url"]
