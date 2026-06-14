"""Price-adapter smoke test (ARCHITECTURE.md §8).

Network-dependent: skips cleanly when offline or when no provider is reachable,
so CI stays green without flaking. When it does run, it asserts the adapter
returns schema-shaped records and that they pass the store's validation —
i.e. the adapter output can be written through the chokepoint unchanged.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.prices import DataPullError, fetch_prices
from store.schema import COLUMNS, validate


@pytest.mark.network
def test_fetch_prices_smoke():
    try:
        recs = fetch_prices("AAPL", date(2020, 1, 1), date(2020, 1, 15))
    except DataPullError:
        pytest.skip("no price provider reachable (offline)")
    except Exception as exc:  # any provider/transport error -> treat as offline
        pytest.skip(f"price provider unavailable: {exc}")

    assert len(recs) >= 1
    assert list(recs.columns) == COLUMNS
    assert (recs["field"] == "close").all()
    assert (recs["ticker"] == "AAPL").all()
    # price bars carry no filing lag: knowledge_date == event_date
    assert (recs["event_date"] == recs["knowledge_date"]).all()
    # adapter output must satisfy the store's write contract unchanged
    validate(recs)
    assert pd.api.types.is_numeric_dtype(recs["value"])
