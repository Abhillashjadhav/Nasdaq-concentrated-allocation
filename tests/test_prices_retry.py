"""Price-adapter resilience tests (ARCHITECTURE.md §8; CLAUDE.md fail-loud).

Exercises retry + exponential backoff + provider fallback + fail-loud quarantine
with injected fake providers and a fake sleep — no network, fully deterministic.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data import prices
from data.prices import DataPullError, fetch_prices


def _ok_frame():
    return pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-02")], "close": [300.0]}
    )


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_yf(ticker, start, end):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limited")
        return _ok_frame(), "yfinance"

    slept = []
    monkeypatch.setattr(prices, "_from_yfinance", flaky_yf)
    recs = fetch_prices(
        "AAPL", date(2020, 1, 1), date(2020, 1, 3),
        retries=3, base_delay=1.0, sleep=slept.append,
    )
    assert calls["n"] == 3
    assert len(recs) == 1 and recs.iloc[0]["source"] == "yfinance"
    assert slept == [1.0, 2.0]  # exponential backoff between the 3 attempts


def test_falls_back_to_stooq(monkeypatch):
    monkeypatch.setattr(prices, "_from_yfinance", lambda *a: (None, "yfinance"))
    monkeypatch.setattr(prices, "_from_stooq", lambda *a: (_ok_frame(), "stooq"))
    recs = fetch_prices(
        "AAPL", date(2020, 1, 1), date(2020, 1, 3), sleep=lambda *_: None
    )
    assert recs.iloc[0]["source"] == "stooq"


def test_exhaustion_raises_and_quarantines(monkeypatch):
    monkeypatch.setattr(prices, "_from_yfinance", lambda *a: (None, "yfinance"))
    monkeypatch.setattr(prices, "_from_stooq", lambda *a: (None, "stooq"))
    with pytest.raises(DataPullError) as exc:
        fetch_prices(
            "ZZZZ", date(2020, 1, 1), date(2020, 1, 3),
            retries=2, sleep=lambda *_: None,
        )
    assert set(exc.value.quarantine) == {"yfinance", "stooq"}  # both named, not dropped
