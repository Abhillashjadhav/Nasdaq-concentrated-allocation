"""Winner-labeler tests (ARCHITECTURE.md §3, §9.10).

The labeler is the sanctioned look-ahead (answer key). These prove: a clear
winner, a clear non-winner, the >=-inclusive boundary rule, a delisted-to-zero
name graded as the -100% disaster (not winner), and an incomplete forward window
flagged as outcome-not-yet-known rather than labeled prematurely.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest.labels import BENCHMARK_TICKER, label_winner
from store.store import PITStore

AS_OF = date(2018, 1, 1)           # horizon_end = 2019-01-01
ENTRY_DATE = "2017-12-29"          # last trading day before as_of
EXIT_DATE = "2018-12-31"           # last trading day before horizon_end


def _close(ticker, event_date, value) -> dict:
    # price bars: knowledge_date == event_date (the trading date), per the store model
    return {"ticker": ticker, "field": "close", "value": float(value),
            "event_date": pd.Timestamp(event_date),
            "knowledge_date": pd.Timestamp(event_date), "source": "fixture"}


@pytest.fixture
def store(tmp_path):
    s = PITStore(tmp_path / "labels.sqlite")
    rows = [
        _close(BENCHMARK_TICKER, ENTRY_DATE, 100), _close(BENCHMARK_TICKER, EXIT_DATE, 110),  # +10%
        _close("WIN", ENTRY_DATE, 100), _close("WIN", EXIT_DATE, 130),    # +30%
        _close("NOPE", ENTRY_DATE, 100), _close("NOPE", EXIT_DATE, 112),  # +12%
        # DEAD delists mid-window: last close 2018-06-29, nothing near horizon_end
        _close("DEAD", ENTRY_DATE, 100), _close("DEAD", "2018-06-29", 50),
    ]
    s.put_data(pd.DataFrame(rows))
    return s


def test_winner_beats_benchmark_by_15pp(store):
    res = label_winner("WIN", AS_OF, store=store)
    assert res.outcome_known is True
    assert res.is_winner is True
    assert res.components["excess_return"] == pytest.approx(0.20, abs=1e-9)


def test_not_winner_below_threshold(store):
    res = label_winner("NOPE", AS_OF, store=store)
    assert res.outcome_known is True
    assert res.is_winner is False
    assert res.components["excess_return"] == pytest.approx(0.02, abs=1e-9)


def test_boundary_is_inclusive(tmp_path):
    """An excess of exactly the threshold is a WINNER (>=, not >). Tested at an
    exactly-representable threshold (0.25) so the assertion targets the rule, not
    float noise (0.15 is not exactly representable in binary)."""
    s = PITStore(tmp_path / "edge.sqlite")
    s.put_data(pd.DataFrame([
        _close(BENCHMARK_TICKER, ENTRY_DATE, 100), _close(BENCHMARK_TICKER, EXIT_DATE, 100),  # flat
        _close("EDGE", ENTRY_DATE, 100), _close("EDGE", EXIT_DATE, 125),   # +25% exactly
        _close("UNDER", ENTRY_DATE, 100), _close("UNDER", EXIT_DATE, 124),  # +24%
    ]))
    at_threshold = label_winner("EDGE", AS_OF, store=s, threshold_pp=25.0)
    assert at_threshold.components["excess_return"] == pytest.approx(0.25, abs=1e-12)
    assert at_threshold.is_winner is True            # exactly at threshold -> winner

    below = label_winner("UNDER", AS_OF, store=s, threshold_pp=25.0)
    assert below.is_winner is False                  # just under -> not


def test_delisted_to_zero_is_disaster_not_winner(store):
    res = label_winner("DEAD", AS_OF, store=store)
    assert res.outcome_known is True                 # benchmark elapsed -> outcome IS known
    assert res.components["delisted"] is True
    assert res.components["exit_price"] == 0.0
    assert res.components["ticker_return"] == pytest.approx(-1.0, abs=1e-12)  # -100%
    assert res.is_winner is False


def test_incomplete_window_not_yet_known(store):
    # horizon_end = 2019-06-01, but the benchmark series stops at 2018-12-31
    res = label_winner("WIN", date(2018, 6, 1), store=store)
    assert res.outcome_known is False
    assert res.is_winner is None
    assert res.reason.startswith("outcome_not_yet_known")
