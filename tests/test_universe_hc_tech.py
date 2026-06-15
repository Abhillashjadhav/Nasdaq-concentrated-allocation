"""Universe classification eval (ARCHITECTURE.md §1, §2.2, §6, §8).

A known tech ticker -> technology; a known healthcare ticker -> healthcare; an
unrelated (bank) ticker -> excluded. SIC is cached in the store and membership is
point-in-time (a name appears only if it was filing as-of the date). Offline via a
fake EDGAR client.
"""

from __future__ import annotations

from datetime import date

import pytest

from data.edgar_client import CikResolver
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
    def get_json(self, path):
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
