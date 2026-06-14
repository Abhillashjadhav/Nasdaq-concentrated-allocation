"""Universe + liquidity tests (ARCHITECTURE.md §2, §7, §9.4).

Covers: as-of determinism, a delisted name present at an earlier date (and gone
later), price/ADV floor exclusions, sector filtering behind the interface,
coverage-gap surfacing, and no-future-leak (reusing the PR-3 no-peek assertion).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from store.store import PITStore
from universe.universe import build_universe, get_universe

CANDIDATES = ["AAPL", "ENRNQ", "PENNY", "THIN", "NOVOL", "ENERGYCO", "GHOST"]


class DictSectorClassifier:
    """Test double for the SectorClassifier interface."""

    def __init__(self, mapping):
        self.mapping = mapping

    def classify(self, ticker, as_of):
        return self.mapping.get(ticker)


SECTORS = DictSectorClassifier(
    {
        "AAPL": "technology", "ENRNQ": "healthcare", "PENNY": "technology",
        "THIN": "technology", "NOVOL": "technology", "ENERGYCO": "energy",
    }
)


def _rows(ticker, field, start, end, value, freq="7D"):
    dates = pd.date_range(start, end, freq=freq)
    return pd.DataFrame(
        {
            "ticker": ticker, "field": field, "value": float(value),
            "event_date": dates, "knowledge_date": dates, "source": "fixture",
        }
    )


@pytest.fixture
def store(tmp_path):
    s = PITStore(tmp_path / "uni.sqlite")
    frames = [
        # liquid tech name, listed 2015->2021
        _rows("AAPL", "close", "2015-01-01", "2021-01-01", 150.0),
        _rows("AAPL", "dollar_volume", "2015-01-01", "2021-01-01", 5e8),
        # delisted healthcare name: data only 2006->2010 (gone after)
        _rows("ENRNQ", "close", "2006-01-01", "2010-06-01", 80.0),
        _rows("ENRNQ", "dollar_volume", "2006-01-01", "2010-06-01", 3e8),
        # below price floor
        _rows("PENNY", "close", "2015-01-01", "2021-01-01", 2.0),
        _rows("PENNY", "dollar_volume", "2015-01-01", "2021-01-01", 5e8),
        # below ADV floor
        _rows("THIN", "close", "2015-01-01", "2021-01-01", 50.0),
        _rows("THIN", "dollar_volume", "2015-01-01", "2021-01-01", 1e5),
        # price present but NO dollar_volume -> ADV coverage gap
        _rows("NOVOL", "close", "2015-01-01", "2021-01-01", 50.0),
        # liquid but wrong sector
        _rows("ENERGYCO", "close", "2015-01-01", "2021-01-01", 50.0),
        _rows("ENERGYCO", "dollar_volume", "2015-01-01", "2021-01-01", 5e8),
        # GHOST: no data at all -> no_price_coverage
    ]
    s.put_data(pd.concat(frames, ignore_index=True))
    return s


def _reasons(result):
    return {e["ticker"]: e["reason"] for e in result.excluded}


def test_universe_2020_filters(store):
    res = build_universe(store=store, as_of=date(2020, 6, 1), candidates=CANDIDATES,
                         sector_classifier=SECTORS)
    assert res.tickers == ["AAPL"]
    r = _reasons(res)
    assert r["ENRNQ"] == "not_listed_as_of"      # delisted -> stale as of 2020
    assert r["PENNY"] == "below_price_floor"
    assert r["THIN"] == "below_adv_floor"
    assert r["NOVOL"] == "adv_coverage_gap"
    assert r["ENERGYCO"] == "sector_excluded"
    assert r["GHOST"] == "no_price_coverage"


def test_delisted_name_present_at_earlier_date(store):
    res = build_universe(store=store, as_of=date(2008, 6, 1), candidates=CANDIDATES,
                         sector_classifier=SECTORS)
    # the dead name is in the historical universe; the survivors (2015+) aren't yet listed
    assert res.tickers == ["ENRNQ"]
    assert _reasons(res)["AAPL"] == "no_price_coverage"


def test_as_of_determinism(store):
    a = build_universe(store=store, as_of=date(2020, 6, 1), candidates=CANDIDATES,
                       sector_classifier=SECTORS)
    b = build_universe(store=store, as_of=date(2020, 6, 1), candidates=CANDIDATES,
                       sector_classifier=SECTORS)
    assert a.tickers == b.tickers
    assert a.excluded == b.excluded


def test_sector_filter_disabled_without_classifier(store):
    res = build_universe(store=store, as_of=date(2020, 6, 1), candidates=CANDIDATES)
    assert res.sector_filter_applied is False
    assert "ENERGYCO" in res.tickers  # not excluded on sector when filter is off


def test_coverage_gaps_surfaced_not_silent(store):
    res = build_universe(store=store, as_of=date(2020, 6, 1), candidates=CANDIDATES,
                         sector_classifier=SECTORS)
    gap_tickers = {g["ticker"] for g in res.coverage_gaps}
    assert {"GHOST", "NOVOL"} <= gap_tickers


def test_no_future_leak_over_universe(store):
    as_ofs = [date(2008, 6, 1), date(2012, 1, 1), date(2020, 6, 1)]
    assert_no_future_rows(store, ["close", "dollar_volume"], CANDIDATES, as_ofs)


def test_get_universe_requires_candidates():
    with pytest.raises(ValueError):
        get_universe(date(2020, 6, 1))
