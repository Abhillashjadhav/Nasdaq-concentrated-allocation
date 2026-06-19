"""Universe classification eval (ARCHITECTURE.md §1, §2.2, §6, §8).

A known tech ticker -> technology; a known healthcare ticker -> healthcare; an
unrelated (bank) ticker -> excluded. SIC is cached in the store and membership is
point-in-time (a name appears only if it was filing as-of the date). Offline via a
fake EDGAR client.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.edgar_client import CikResolver, EdgarHTTPError
from universe.hc_tech import SIC_FIELD, classify_and_cache, nasdaq_hc_tech_universe
from universe.sic import HEALTHCARE, TECHNOLOGY, classify_sic
from store.store import PITStore

COMPANY_TICKERS = {
    "0": {"cik_str": 1, "ticker": "TECHX"},
    "1": {"cik_str": 2, "ticker": "HEALX"},
    "2": {"cik_str": 3, "ticker": "BANKX"},
    "3": {"cik_str": 4, "ticker": "NOSIC"},
}


def _subs(sic, first="2015-03-01"):
    return {"sic": sic, "filings": {"recent": {"filingDate": [first, "2016-05-01"]}}}


SUBMISSIONS = {
    "0000000001": _subs("7372"),   # prepackaged software -> technology
    "0000000002": _subs("2836"),   # biological products  -> healthcare
    "0000000003": _subs("6022"),   # state commercial bank -> excluded
    "0000000004": _subs(""),       # no SIC -> quarantine
}


class FakeClient:
    def __init__(self):
        self.calls = 0  # every EDGAR get_json (company_tickers + submissions)

    def get_json(self, path):
        self.calls += 1
        if "company_tickers" in path:
            return COMPANY_TICKERS
        for cik, data in SUBMISSIONS.items():
            if cik in path:
                return data
        raise KeyError(path)


def test_classify_sic_known_codes():
    assert classify_sic(7372) == TECHNOLOGY      # software
    assert classify_sic(3674) == TECHNOLOGY      # semiconductors
    assert classify_sic(2836) == HEALTHCARE      # biologics
    assert classify_sic(8011) == HEALTHCARE      # offices of doctors
    assert classify_sic(6022) is None            # bank -> excluded
    assert classify_sic(None) is None
    assert classify_sic("not-a-number") is None


def test_classify_and_cache_builds_tech_healthcare_universe(tmp_path):
    store = PITStore(tmp_path / "uni.sqlite")
    fake = FakeClient()
    symbols = ["TECHX", "HEALX", "BANKX", "NOSIC", "ZZZZ"]  # ZZZZ unknown ticker
    res = classify_and_cache(symbols, client=fake, resolver=CikResolver(fake), store=store)

    members = nasdaq_hc_tech_universe(date(2099, 1, 1), symbols, store=store)
    assert set(members) == {"TECHX", "HEALX"}    # bank excluded; no-sic + unknown not classified
    assert res.n_classified == 3                 # TECHX, HEALX, BANKX all fetched + cached
    assert res.n_kept == 2                        # of those, only tech + healthcare are in-scope
    assert res.n_quarantined == 2                # NOSIC (no sic) + ZZZZ (unknown ticker)
    assert any(g["ticker"] == "NOSIC" for g in res.quarantine)


def test_membership_is_point_in_time(tmp_path):
    store = PITStore(tmp_path / "pit.sqlite")
    fake = FakeClient()
    classify_and_cache(["TECHX"], client=fake, resolver=CikResolver(fake), store=store)
    # TECHX's earliest filing is 2015-03-01 -> not yet "filing" in 2014
    assert nasdaq_hc_tech_universe(date(2014, 1, 1), ["TECHX"], store=store) == []
    assert nasdaq_hc_tech_universe(date(2016, 1, 1), ["TECHX"], store=store) == ["TECHX"]


def test_classification_is_cached_not_refetched(tmp_path):
    store = PITStore(tmp_path / "cache.sqlite")
    fake = FakeClient()
    classify_and_cache(["TECHX"], client=fake, resolver=CikResolver(fake), store=store)
    again = classify_and_cache(["TECHX"], client=fake, resolver=CikResolver(fake), store=store)
    assert again.n_cached == 1 and again.n_classified == 0  # skipped, already cached


# --- execution-robustness evals (scaling to ~3000 tickers) -------------------

def test_resumability_no_edgar_calls_for_cached_tickers(tmp_path):
    """The build is a resumable, one-time cost: after caching, a re-run makes ZERO
    EDGAR calls for already-classified tickers."""
    store = PITStore(tmp_path / "resume.sqlite")
    first = FakeClient()
    classify_and_cache(["TECHX", "HEALX"], client=first, resolver=CikResolver(first), store=store)
    assert first.calls > 0  # fetched on the first pass

    second = FakeClient()  # fresh client/resolver -> any EDGAR call would increment
    res = classify_and_cache(["TECHX", "HEALX"], client=second, resolver=CikResolver(second), store=store)
    assert second.calls == 0           # nothing fetched — read entirely from the cache
    assert res.n_cached == 2 and res.n_classified == 0


def test_incremental_cache_persists_before_an_unexpected_error(tmp_path):
    """Each ticker is persisted the moment it is computed (not batched at the end),
    so a crash mid-build does not lose earlier progress."""
    store = PITStore(tmp_path / "inc.sqlite")

    class _Boom(FakeClient):
        def get_json(self, path):
            if "0000000002" in path:   # HEALX submissions -> unexpected (non-quarantined) error
                raise RuntimeError("boom")
            return super().get_json(path)

    fake = _Boom()
    with pytest.raises(RuntimeError):
        classify_and_cache(["TECHX", "HEALX"], client=fake, resolver=CikResolver(fake), store=store)
    # TECHX was written incrementally before HEALX blew up (batch-at-end would lose it)
    assert not store.get_data(SIC_FIELD, "TECHX", date(2099, 1, 1)).empty


def test_refresh_forces_reclassification(tmp_path):
    store = PITStore(tmp_path / "refresh.sqlite")
    first = FakeClient()
    classify_and_cache(["TECHX"], client=first, resolver=CikResolver(first), store=store)
    second = FakeClient()
    res = classify_and_cache(["TECHX"], client=second, resolver=CikResolver(second),
                             store=store, refresh=True)
    assert second.calls > 0  # re-fetched despite the cache
    assert res.n_classified == 1 and res.n_cached == 0


def test_universe_limit_caps_processing(tmp_path):
    store = PITStore(tmp_path / "limit.sqlite")
    fake = FakeClient()
    res = classify_and_cache(["TECHX", "HEALX", "BANKX"], client=fake,
                             resolver=CikResolver(fake), store=store, limit=1)
    assert res.n_symbols == 1  # only the first symbol processed
    assert nasdaq_hc_tech_universe(date(2099, 1, 1), ["TECHX", "HEALX", "BANKX"], store=store) == ["TECHX"]


def test_progress_logging(tmp_path, caplog):
    import logging
    store = PITStore(tmp_path / "prog.sqlite")
    fake = FakeClient()
    with caplog.at_level(logging.INFO):
        classify_and_cache(["TECHX", "HEALX"], client=fake, resolver=CikResolver(fake),
                           store=store, log_every=1)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("classified" in m and "kept" in m and "failed" in m for m in msgs)


# --- BUG 1: a failed/empty fetch must NOT be cached as a skip verdict ----------
# A poisoned cache (failures written as negative verdicts) makes every later run
# read "kept 0" straight from the store. The fix: only a genuine classification is
# cached; a fetch that errors or returns empty stays uncached so a re-run retries.

class _FailingThenGood:
    """submissions fetch errors while ``fail`` is set; company_tickers always works."""

    def __init__(self, fail):
        self.fail = fail
        self.calls = 0

    def get_json(self, path):
        self.calls += 1
        if "company_tickers" in path:
            return COMPANY_TICKERS
        if self.fail:
            raise EdgarHTTPError("simulated network/SSL failure")
        for cik, data in SUBMISSIONS.items():
            if cik in path:
                return data
        raise KeyError(path)


def test_failed_fetch_is_not_cached_and_is_retried(tmp_path):
    store = PITStore(tmp_path / "poison.sqlite")
    # Run 1: every submissions fetch fails -> quarantined, NOTHING written to the store.
    bad = _FailingThenGood(fail=True)
    r1 = classify_and_cache(["TECHX"], client=bad, resolver=CikResolver(bad),
                            store=store, sleep=lambda _s: None)
    assert r1.n_quarantined == 1 and r1.n_classified == 0
    assert store.get_data(SIC_FIELD, "TECHX", date(2100, 1, 1)).empty  # not cached as a skip

    # Run 2: same store, fetch now works -> the ticker is RE-FETCHED, not stuck "skipped".
    good = _FailingThenGood(fail=False)
    r2 = classify_and_cache(["TECHX"], client=good, resolver=CikResolver(good), store=store)
    assert good.calls > 0                       # genuinely retried, not read from a poisoned cache
    assert r2.n_classified == 1 and r2.n_cached == 0
    assert nasdaq_hc_tech_universe(date(2099, 1, 1), ["TECHX"], store=store) == ["TECHX"]


def test_empty_submissions_not_cached(tmp_path):
    """A soft-empty response (200 with no SIC/filings) is quarantined, never written
    as a skip verdict — so it is retried on the next run rather than poisoning cache."""
    store = PITStore(tmp_path / "empty.sqlite")

    class _Empty:
        def get_json(self, path):
            if "company_tickers" in path:
                return COMPANY_TICKERS
            return {}  # no sic, no filings

    e = _Empty()
    res = classify_and_cache(["TECHX"], client=e, resolver=CikResolver(e), store=store)
    assert res.n_quarantined == 1 and res.n_classified == 0
    assert store.get_data(SIC_FIELD, "TECHX", date(2100, 1, 1)).empty  # retryable, not poisoned


# --- BUG 2: a hanging EDGAR fetch must time out, count as failed, and NOT block ---

def test_fetch_timeout_increments_failed_and_continues(tmp_path):
    import threading
    store = PITStore(tmp_path / "timeout.sqlite")

    class _Hang:
        """TECHX's submissions fetch blocks past the deadline (simulating an SSL read
        that never returns); the rest resolve normally."""

        def __init__(self):
            self.gate = threading.Event()  # never set -> the worker thread parks

        def get_json(self, path):
            if "company_tickers" in path:
                return COMPANY_TICKERS
            if "0000000001" in path:        # TECHX -> hang
                self.gate.wait(30)
                return _subs("7372")
            for cik, data in SUBMISSIONS.items():
                if cik in path:
                    return data
            raise KeyError(path)

    h = _Hang()
    try:
        # Tiny deadline + no retry sleeps keep the test fast; HEALX must still classify.
        res = classify_and_cache(
            ["TECHX", "HEALX"], client=h, resolver=CikResolver(h), store=store,
            fetch_timeout=0.1, fetch_retries=1, sleep=lambda _s: None,
        )
        assert res.n_quarantined == 1                          # TECHX timed out -> failed
        assert any(g["ticker"] == "TECHX" for g in res.quarantine)
        assert res.n_kept == 1                                 # HEALX classified -> loop continued
        assert nasdaq_hc_tech_universe(date(2099, 1, 1), ["TECHX", "HEALX"], store=store) == ["HEALX"]
        assert store.get_data(SIC_FIELD, "TECHX", date(2100, 1, 1)).empty  # timeout not cached
    finally:
        h.gate.set()  # release the parked worker thread


def test_progress_streams_to_stdout(tmp_path, capsys):
    """Progress must reach STDOUT (not only the logging module) so it streams
    through the user's `tee` — a startup line before the first EDGAR call and a
    per-batch progress line."""
    store = PITStore(tmp_path / "stdout.sqlite")
    fake = FakeClient()
    classify_and_cache(["TECHX", "HEALX"], client=fake, resolver=CikResolver(fake),
                       store=store, log_every=1)
    out = capsys.readouterr().out
    assert "building universe: 2 symbols to classify" in out  # startup, no silent gap
    assert "cached: 0, to fetch: 2" in out
    assert "classified 1/2" in out and "cache-hits" in out     # a progress line streamed
    assert "universe build complete:" in out                   # completion line
