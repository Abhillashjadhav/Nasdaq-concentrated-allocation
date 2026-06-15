"""FRED macro adapter tests (ARCHITECTURE.md §5, §6).

Golden parse from a hand-built FRED observations fixture (no network), the PIT
publication-lag mapping, missing-value skipping, coverage-gap flagging, fail-loud
on a missing API key, field alignment with macro.regime, and a network-marked
live smoke that skips offline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.fred import (
    API_KEY_ENV,
    MACRO_TICKER,
    SERIES,
    FredClient,
    FredConfigError,
    FredHTTPError,
    fetch_macro,
)
from store.store import PITStore


class FakeFredClient:
    """Returns canned observations JSON by series id; no network."""

    def __init__(self, by_series):
        self.by_series = by_series
        self.calls = []

    def get_observations(self, series_id, *, observation_start=None):
        self.calls.append(series_id)
        return self.by_series.get(series_id, {"observations": []})


def _obs(rows):
    return {"observations": [{"date": d, "value": v} for d, v in rows]}


def test_golden_parse_and_pit_lag():
    fake = FakeFredClient({
        "BAMLH0A0HYM2": _obs([("2020-03-30", "8.00"), ("2020-03-31", ".")]),  # "." skipped
    })
    res = fetch_macro(client=fake, fields=["hy_oas"])

    assert res.gaps == []
    assert len(res.records) == 1  # the "." missing value was skipped, not zeroed
    row = res.records.iloc[0]
    assert row["ticker"] == MACRO_TICKER and row["field"] == "hy_oas"
    assert row["value"] == 8.0 and row["source"] == "fred"
    # event_date = observation date; knowledge_date = + 1d publication lag (PIT)
    assert pd.Timestamp(row["event_date"]) == pd.Timestamp("2020-03-30")
    assert pd.Timestamp(row["knowledge_date"]) == pd.Timestamp("2020-03-31")


def test_writes_through_store_and_is_point_in_time(tmp_path):
    fake = FakeFredClient({"VIXCLS": _obs([("2020-03-30", "60.0")])})
    store = PITStore(tmp_path / "fred.sqlite")
    res = fetch_macro(client=fake, fields=["vix"], store=store, write=True)
    assert res.n_written == 1

    # observed 03-30, released 03-31: invisible as-of 03-30, visible 03-31
    assert store.get_data("vix", MACRO_TICKER, date(2020, 3, 30)).empty
    visible = store.get_data("vix", MACRO_TICKER, date(2020, 3, 31))
    assert len(visible) == 1 and visible.iloc[0]["value"] == 60.0


def test_missing_series_flagged_not_zeroed():
    fake = FakeFredClient({"DFF": {"observations": []}})  # series returns nothing
    res = fetch_macro(client=fake, fields=["fed_funds_rate"])
    assert res.records.empty
    assert any(g["field"] == "fed_funds_rate" and g["reason"] == "no_fred_observations"
               for g in res.gaps)


def test_business_day_lag_skips_weekend(tmp_path):
    # a Friday (2020-03-27) observation must not be visible until its Monday release
    fake = FakeFredClient({"VIXCLS": _obs([("2020-03-27", "20.0")])})
    store = PITStore(tmp_path / "bday.sqlite")
    fetch_macro(client=fake, fields=["vix"], store=store, write=True)
    assert store.get_data("vix", MACRO_TICKER, date(2020, 3, 29)).empty       # Sunday
    assert len(store.get_data("vix", MACRO_TICKER, date(2020, 3, 30))) == 1   # Monday


class _KeyLeakSession:
    """Simulates requests surfacing the full URL (with api_key) in the error."""

    def get(self, url, params=None, headers=None, timeout=None):
        full = f"{url}?series_id={params['series_id']}&api_key={params['api_key']}"
        raise RuntimeError(f"500 Server Error for url: {full}")


def test_api_key_never_leaks_into_error():
    c = FredClient(api_key="SECRETKEY123", session=_KeyLeakSession(),
                   retries=1, sleep=lambda *_: None)
    with pytest.raises(FredHTTPError) as ei:
        c.get_observations("BAMLH0A0HYM2")
    assert "SECRETKEY123" not in str(ei.value)        # redacted from the message
    assert ei.value.__cause__ is None                 # chained cause suppressed


def test_missing_api_key_fails_loud(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(FredConfigError):
        FredClient()
    assert FredClient(api_key="abc123").api_key == "abc123"


def test_series_align_with_macro_regime():
    from macro.regime import HY_OAS, MACRO_TICKER as REGIME_MACRO, POLICY_RATE, VIX
    assert set(SERIES) == {HY_OAS, POLICY_RATE, VIX}
    assert MACRO_TICKER == REGIME_MACRO


@pytest.mark.network
def test_live_smoke_skips_offline():
    try:
        res = fetch_macro(client=FredClient(), fields=["vix"], observation_start="2024-01-01")
    except Exception as exc:  # no key / offline / blocked -> skip, don't fail CI
        pytest.skip(f"FRED unavailable: {exc}")
    assert len(res.records) >= 1
