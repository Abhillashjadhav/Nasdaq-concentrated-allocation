"""Point-in-time store tests (ARCHITECTURE.md §6, §8).

Guards the no-peek chokepoint and the fail-loud write path that the rest of the
harness is built on. These land WITH the store logic, per the evals-first rule.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from store.store import PITStore


def _store(tmp_path) -> PITStore:
    return PITStore(tmp_path / "pit.sqlite")


def _record(**over) -> pd.DataFrame:
    row = {
        "ticker": "AAPL",
        "field": "close",
        "value": 100.0,
        "event_date": pd.Timestamp("2020-06-15"),
        "knowledge_date": pd.Timestamp("2020-06-15"),
        "source": "test",
    }
    row.update(over)
    return pd.DataFrame([row])


def test_no_peek_filters_future_rows(tmp_path):
    """A row known on 2020-06-15 is invisible as of 2020-06-14, visible on 06-15."""
    store = _store(tmp_path)
    store.put_data(_record(knowledge_date=pd.Timestamp("2020-06-15")))

    assert store.get_data("close", "AAPL", date(2020, 6, 14)).empty
    visible = store.get_data("close", "AAPL", date(2020, 6, 15))
    assert len(visible) == 1
    assert visible.iloc[0]["value"] == 100.0


def test_get_data_newest_event_date_first(tmp_path):
    store = _store(tmp_path)
    store.put_data(
        pd.concat(
            [
                _record(event_date=pd.Timestamp("2020-01-02"), value=1.0),
                _record(event_date=pd.Timestamp("2020-03-02"), value=3.0),
                _record(event_date=pd.Timestamp("2020-02-02"), value=2.0),
            ],
            ignore_index=True,
        )
    )
    out = store.get_data("close", "AAPL", date(2020, 6, 15))
    assert list(out["value"]) == [3.0, 2.0, 1.0]


def test_malformed_record_raises_on_put(tmp_path):
    """Validation must fail loud — a missing required column is never silently dropped."""
    store = _store(tmp_path)
    bad = _record().drop(columns=["source"])
    with pytest.raises(SchemaErrors):
        store.put_data(bad)


def test_wrong_type_value_raises_on_put(tmp_path):
    store = _store(tmp_path)
    bad = _record(value="not-a-number")
    with pytest.raises(SchemaErrors):
        store.put_data(bad)
