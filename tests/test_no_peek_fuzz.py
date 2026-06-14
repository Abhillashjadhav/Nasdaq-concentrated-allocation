"""Hard no-peek fuzz (ARCHITECTURE.md §2, §8).

Populates a store with records spread across many knowledge_dates, then fuzzes
a grid of (field, ticker, as_of) probes and asserts NO probe ever returns a
future row. Also proves the checker itself fails loud on a deliberately leaky
store — a guard that always passes would be worthless.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from evals.no_peek import LeakError, assert_no_future_rows
from store.store import PITStore

TICKERS = ["AAPL", "MSFT", "ENRNQ"]  # incl. a delisted name


def _populate(tmp_path) -> PITStore:
    store = PITStore(tmp_path / "fuzz.sqlite")
    rows = []
    for t in TICKERS:
        for i in range(24):  # two years of monthly rows
            kd = pd.Timestamp("2018-01-15") + pd.DateOffset(months=i)
            rows.append(
                {
                    "ticker": t, "field": "close", "value": 100.0 + i,
                    "event_date": kd, "knowledge_date": kd, "source": "fixture",
                }
            )
    store.put_data(pd.DataFrame(rows))
    return store


def test_no_future_rows_across_fuzzed_grid(tmp_path):
    store = _populate(tmp_path)
    as_ofs = [date(2018, 1, 1) + timedelta(days=37 * k) for k in range(24)]
    probes = assert_no_future_rows(store, ["close"], TICKERS, as_ofs)
    assert probes == len(TICKERS) * len(as_ofs)


class _LeakyStore:
    """A store that violates the contract — used to prove the checker bites."""

    def get_data(self, field, ticker, as_of):
        future = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
        return pd.DataFrame(
            [{"ticker": ticker, "field": field, "value": 1.0,
              "event_date": future, "knowledge_date": future, "source": "leak"}]
        )


def test_checker_detects_a_leak():
    with pytest.raises(LeakError):
        assert_no_future_rows(_LeakyStore(), ["close"], ["AAPL"], [date(2020, 6, 14)])
