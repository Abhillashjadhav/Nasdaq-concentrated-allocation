"""Filing-lag test (ARCHITECTURE.md §2, §6, §8).

Period-end data must not be visible until its filing knowledge_date. Here a Q1
figure with event_date 2020-03-31 (period end) only becomes public on its 10-Q
filing date 2020-05-15 (~45d later). Between those dates the value is invisible,
even though the period has ended.

Uses a fixture for the lag. Real EDGAR lag wiring (computing knowledge_date from
the actual filing) lands with the EDGAR adapter PR; this proves the store
honours whatever knowledge_date the ingest layer assigns.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from store.store import PITStore

PERIOD_END = pd.Timestamp("2020-03-31")  # event_date (the quarter it describes)
FILING_DATE = pd.Timestamp("2020-05-15")  # knowledge_date (~45d 10-Q lag)


def _store_with_lagged_fact(tmp_path) -> PITStore:
    store = PITStore(tmp_path / "lag.sqlite")
    store.put_data(
        pd.DataFrame(
            [{
                "ticker": "AAPL", "field": "eps", "value": 2.55,
                "event_date": PERIOD_END, "knowledge_date": FILING_DATE,
                "source": "fixture",
            }]
        )
    )
    return store


def test_period_end_invisible_until_filing(tmp_path):
    store = _store_with_lagged_fact(tmp_path)
    # period has ended (after 03-31) but the 10-Q is not yet filed -> invisible
    assert store.get_data("eps", "AAPL", date(2020, 4, 15)).empty
    # day before filing -> still invisible
    assert store.get_data("eps", "AAPL", date(2020, 5, 14)).empty


def test_visible_on_and_after_filing(tmp_path):
    store = _store_with_lagged_fact(tmp_path)
    on_filing = store.get_data("eps", "AAPL", date(2020, 5, 15))
    assert len(on_filing) == 1
    assert on_filing.iloc[0]["value"] == 2.55
    # event_date still reflects the period described, distinct from knowledge_date
    assert pd.Timestamp(on_filing.iloc[0]["event_date"]) == PERIOD_END
    assert not store.get_data("eps", "AAPL", date(2021, 1, 1)).empty
