"""EDGAR fundamentals adapter tests (ARCHITECTURE.md §5, §6).

Golden parse from a hand-built mini companyfacts (no network), fallback tags,
coverage-gap flagging, CIK resolution + fail-loud unknown ticker, missing
User-Agent fail-loud, throttle + retry units, point-in-time restatement handling,
quality-field alignment, and a network-marked live smoke that skips offline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from data.edgar_client import (
    EdgarClient,
    EdgarConfigError,
    MIN_INTERVAL_SECONDS,
    UnknownTickerError,
    USER_AGENT_ENV,
    CikResolver,
)
from data.fundamentals import CONCEPTS, fetch_fundamentals
from store.store import PITStore

COMPANY_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


def _usd(end, val, filed):
    return {"end": end, "val": val, "filed": filed, "form": "10-K", "fy": 2020, "fp": "FY"}


def _facts(usgaap):
    return {"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": usgaap}}


GOLDEN_VALUES = {
    "NetIncomeLoss": ("net_income", 150, "USD"),
    "NetCashProvidedByUsedInOperatingActivities": ("cfo", 200, "USD"),
    "Assets": ("total_assets", 1100, "USD"),
    "LongTermDebtNoncurrent": ("long_term_debt", 300, "USD"),
    "AssetsCurrent": ("current_assets", 500, "USD"),
    "LiabilitiesCurrent": ("current_liabilities", 200, "USD"),
    "CommonStockSharesOutstanding": ("shares_outstanding", 60, "shares"),
    "GrossProfit": ("gross_profit", 480, "USD"),
    "Revenues": ("revenue", 1000, "USD"),
}


def _golden_companyfacts():
    usgaap = {}
    for tag, (_field, val, unit) in GOLDEN_VALUES.items():
        usgaap[tag] = {"units": {unit: [_usd("2020-12-31", val, "2021-02-26")]}}
    return _facts(usgaap)


class FakeClient:
    """Returns canned JSON by URL substring; no network."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path_or_url):
        self.calls.append(path_or_url)
        for key, value in self.responses.items():
            if key in path_or_url:
                return value
        raise KeyError(path_or_url)


def _fake(companyfacts):
    fake = FakeClient({"company_tickers": COMPANY_TICKERS, "companyfacts": companyfacts})
    return fake, CikResolver(fake)


# --- CIK resolution -----------------------------------------------------------

def test_cik_resolution_zero_pads():
    fake, resolver = _fake(_golden_companyfacts())
    assert resolver.resolve("AAPL") == "0000320193"
    assert resolver.resolve("msft") == "0000789019"  # case-insensitive


def test_unknown_ticker_fails_loud():
    _, resolver = _fake(_golden_companyfacts())
    with pytest.raises(UnknownTickerError):
        resolver.resolve("ZZZZ")


def test_company_tickers_fetched_once_and_cached():
    fake, resolver = _fake(_golden_companyfacts())
    resolver.resolve("AAPL")
    resolver.resolve("MSFT")
    assert sum("company_tickers" in c for c in fake.calls) == 1  # cached


# --- golden parse -------------------------------------------------------------

def test_golden_parse_extracts_nine_concepts():
    fake, resolver = _fake(_golden_companyfacts())
    res = fetch_fundamentals("AAPL", client=fake, resolver=resolver)

    assert res.cik == "0000320193"
    assert res.gaps == []
    expected = {field: val for (field, val, _u) in GOLDEN_VALUES.values()}
    assert set(res.records["field"]) == set(expected)
    for field, val in expected.items():
        row = res.records[res.records["field"] == field]
        assert len(row) == 1
        assert row.iloc[0]["value"] == float(val)
        assert pd.Timestamp(row.iloc[0]["event_date"]) == pd.Timestamp("2020-12-31")
        assert pd.Timestamp(row.iloc[0]["knowledge_date"]) == pd.Timestamp("2021-02-26")
        assert row.iloc[0]["source"] == "edgar"


def test_fallback_tag_used_when_primary_absent():
    # net income only under the fallback tag ProfitLoss; revenue under SalesRevenueNet
    usgaap = {
        "ProfitLoss": {"units": {"USD": [_usd("2020-12-31", 42, "2021-02-26")]}},
        "SalesRevenueNet": {"units": {"USD": [_usd("2020-12-31", 900, "2021-02-26")]}},
    }
    fake, resolver = _fake(_facts(usgaap))
    res = fetch_fundamentals("AAPL", client=fake, resolver=resolver)
    assert res.records.set_index("field").loc["net_income", "value"] == 42.0
    assert res.records.set_index("field").loc["revenue", "value"] == 900.0


def test_missing_concept_flagged_as_gap_not_zeroed():
    facts = _golden_companyfacts()
    del facts["facts"]["us-gaap"]["GrossProfit"]  # company exposes no gross-profit tag
    fake, resolver = _fake(facts)
    res = fetch_fundamentals("AAPL", client=fake, resolver=resolver)

    assert "gross_profit" not in set(res.records["field"])     # not silently zeroed
    assert any(g["field"] == "gross_profit" and g["reason"] == "no_edgar_concept"
               for g in res.gaps)


# --- restatement / point-in-time ---------------------------------------------

def test_restatement_as_of_original_returns_original(tmp_path):
    usgaap = {"NetIncomeLoss": {"units": {"USD": [
        _usd("2020-12-31", 150, "2021-02-26"),   # original 10-K
        _usd("2020-12-31", 130, "2021-08-01"),   # later restatement (amendment)
    ]}}}
    fake, resolver = _fake(_facts(usgaap))
    store = PITStore(tmp_path / "edgar.sqlite")
    res = fetch_fundamentals("AAPL", client=fake, resolver=resolver, store=store, write=True)

    ni = res.records[res.records["field"] == "net_income"]
    assert len(ni) == 2 and res.n_written == 2  # both versions kept, not collapsed

    # as-of before the restatement was filed -> only the original is known
    pre = store.get_data("net_income", "AAPL", date(2021, 3, 1))
    assert len(pre) == 1 and pre.iloc[0]["value"] == 150.0
    # later -> both versions are visible
    post = store.get_data("net_income", "AAPL", date(2021, 9, 1))
    assert len(post) == 2


# --- client config / throttle / retry ----------------------------------------

def test_missing_user_agent_fails_loud(monkeypatch):
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    with pytest.raises(EdgarConfigError):
        EdgarClient()
    # explicit contact info is accepted
    assert EdgarClient(user_agent="Jane Doe jane@example.com").user_agent


class _Clock:
    def __init__(self, vals):
        self.vals, self.i = list(vals), 0

    def __call__(self):
        v = self.vals[min(self.i, len(self.vals) - 1)]
        self.i += 1
        return v


def test_throttle_waits_between_requests():
    slept = []
    c = EdgarClient(user_agent="t t@e.com", min_interval=0.1, sleep=slept.append,
                    clock=_Clock([100.0, 100.02, 100.02]))
    c._throttle()                      # first call: no wait, records last = 100.0
    c._throttle()                      # 0.02s elapsed < 0.1 -> waits the 0.08 remainder
    assert slept and slept[0] == pytest.approx(0.08, abs=1e-9)


class _FlakySession:
    def __init__(self, fail_times, payload):
        self.fail_times, self.payload, self.n = fail_times, payload, 0

    def get(self, url, headers=None, timeout=None):
        self.n += 1
        if self.n <= self.fail_times:
            raise RuntimeError("transient")
        return _Resp(self.payload)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_get_json_retries_then_succeeds():
    session = _FlakySession(fail_times=2, payload={"ok": True})
    c = EdgarClient(user_agent="t t@e.com", session=session, base_delay=0.0,
                    sleep=lambda *_: None)
    assert c.get_json("/x")["ok"] is True
    assert session.n == 3  # failed twice, succeeded on the third


def test_throttle_default_is_under_10_per_sec():
    assert MIN_INTERVAL_SECONDS >= 0.125  # ~8 req/s, with margin under SEC's ~10/s


class _Resp429:
    def __init__(self, retry_after=None):
        self.status_code = 429
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}

    def raise_for_status(self):
        err = requests.exceptions.HTTPError("429 Too Many Requests")
        err.response = self  # status_code + headers visible to the backoff logic
        raise err


class _Resp200:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}

    @property
    def text(self):
        return "<ownershipDocument/>"


class _Rate429Session:
    """429 (with optional Retry-After) on the first N calls, then 200."""

    def __init__(self, n_429, retry_after=None):
        self.n_429, self.retry_after, self.n = n_429, retry_after, 0

    def get(self, url, headers=None, timeout=None):
        self.n += 1
        return _Resp429(self.retry_after) if self.n <= self.n_429 else _Resp200()


def test_429_is_retryable_and_honors_retry_after():
    slept = []
    session = _Rate429Session(n_429=1, retry_after="2")
    c = EdgarClient(user_agent="t t@e.com", session=session, base_delay=1.0,
                    sleep=slept.append, clock=lambda: 0.0)
    assert c.get_json("/x") == {"ok": True}   # backed off and recovered, not quarantined
    assert session.n == 2                      # one 429, then success
    assert 2.0 in slept                        # honored Retry-After (not exponential 1.0)


def test_429_without_retry_after_uses_exponential_backoff():
    slept = []
    session = _Rate429Session(n_429=1, retry_after=None)
    c = EdgarClient(user_agent="t t@e.com", session=session, base_delay=1.0,
                    sleep=slept.append, clock=lambda: 0.0)
    assert c.get_json("/x") == {"ok": True}
    assert 1.0 in slept                        # exponential fallback (base_delay * 2**0)


def test_throttle_covers_get_text():
    # the .xml path (get_text) goes through the same throttled _get as get_json
    slept = []
    c = EdgarClient(user_agent="t t@e.com", session=_Rate429Session(n_429=0),
                    min_interval=0.125, sleep=slept.append, clock=lambda: 0.0)
    c.get_json("/a")
    c.get_text("/b")
    assert any(s == pytest.approx(0.125, abs=1e-9) for s in slept)  # throttled between calls


# --- alignment + live smoke ---------------------------------------------------

def test_concepts_align_with_quality_fields():
    from signals.quality import FIELDS as QUALITY_FIELDS
    assert sorted(c[0] for c in CONCEPTS) == sorted(QUALITY_FIELDS)


@pytest.mark.network
def test_live_smoke_skips_offline():
    try:
        client = EdgarClient(user_agent="stockscope-test test@example.com")
        res = fetch_fundamentals("AAPL", client=client)
    except Exception as exc:  # offline / blocked / 403 -> skip, don't fail CI
        pytest.skip(f"EDGAR unavailable: {exc}")
    assert len(res.records) >= 1
