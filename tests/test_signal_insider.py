"""Insider cluster-buying tests (ARCHITECTURE.md §4, §8, §9.7).

Golden case: >=3 distinct code-P open-market buys form a cluster, while an M
exercise and an A grant (other fields) are ignored, an out-of-window P buy is
excluded, and a repeat buyer counts once. Plus no-cluster, covered-but-quiet,
no-coverage flag, and no-future-leak (filing lag) guards.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from signals.insider import (
    BUY_FIELD,
    COVERAGE_FIELD,
    insider_cluster_score,
)
from store.store import PITStore


def _txn(ticker, field, insider_id, txn_date, filing_date) -> dict:
    return {"ticker": ticker, "field": field, "value": float(insider_id),
            "event_date": pd.Timestamp(txn_date),
            "knowledge_date": pd.Timestamp(filing_date), "source": "fixture"}


def _covered(ticker, kd="2021-01-01") -> dict:
    return {"ticker": ticker, "field": COVERAGE_FIELD, "value": 1.0,
            "event_date": pd.Timestamp(kd), "knowledge_date": pd.Timestamp(kd),
            "source": "fixture"}


def _store(tmp_path, rows) -> PITStore:
    s = PITStore(tmp_path / "f4.sqlite")
    s.put_data(pd.DataFrame(rows))
    return s


AS_OF = date(2021, 3, 31)


def test_golden_cluster_excludes_non_P_and_out_of_window(tmp_path):
    rows = [
        _covered("CLUS"),
        # three distinct code-P open-market buys inside the 90d window
        _txn("CLUS", BUY_FIELD, 101, "2021-02-01", "2021-02-03"),
        _txn("CLUS", BUY_FIELD, 102, "2021-02-10", "2021-02-12"),
        _txn("CLUS", BUY_FIELD, 103, "2021-02-15", "2021-02-17"),
        # same insider buying again -> counts once toward DISTINCT
        _txn("CLUS", BUY_FIELD, 101, "2021-02-20", "2021-02-22"),
        # an out-of-window P buy (10 months earlier) -> excluded by window
        _txn("CLUS", BUY_FIELD, 106, "2020-06-01", "2020-06-03"),
        # an option exercise (M) and a grant (A) on OTHER fields -> ignored
        _txn("CLUS", "form4_exercise_M", 104, "2021-02-05", "2021-02-07"),
        _txn("CLUS", "form4_grant_A", 105, "2021-02-06", "2021-02-08"),
    ]
    res = insider_cluster_score("CLUS", AS_OF, store=_store(tmp_path, rows))

    # distinct in-window P buyers = {101,102,103} = 3 -> band(3,0,6) = 50.0
    assert res.insufficient_data is False
    assert res.cluster_detected is True
    assert res.components["distinct_buyers"] == 3
    assert res.components["n_buy_txns"] == 4  # 4 in-window P rows (incl. the repeat)
    assert res.score == pytest.approx(50.0, abs=1e-9)


def test_two_buyers_is_not_a_cluster(tmp_path):
    rows = [
        _covered("PAIR"),
        _txn("PAIR", BUY_FIELD, 201, "2021-02-01", "2021-02-03"),
        _txn("PAIR", BUY_FIELD, 202, "2021-02-10", "2021-02-12"),
    ]
    res = insider_cluster_score("PAIR", AS_OF, store=_store(tmp_path, rows))
    assert res.cluster_detected is False
    assert res.components["distinct_buyers"] == 2
    assert res.score == pytest.approx(200.0 / 6.0, abs=1e-9)  # ~33.33, below 50
    assert res.score < 50.0


def test_covered_but_quiet_scores_zero_not_flagged(tmp_path):
    res = insider_cluster_score("QUIET", AS_OF, store=_store(tmp_path, [_covered("QUIET")]))
    assert res.insufficient_data is False  # covered -> scored, not flagged
    assert res.score == pytest.approx(0.0, abs=1e-12)
    assert res.cluster_detected is False


def test_no_coverage_is_flagged_not_scored(tmp_path):
    # DARK has no coverage marker at all -> flagged, never scored 0
    store = _store(tmp_path, [_covered("OTHER")])
    res = insider_cluster_score("DARK", AS_OF, store=store)
    assert res.insufficient_data is True
    assert res.score is None
    assert res.reason == "no_form4_coverage"


def test_no_future_leak_respects_filing_lag(tmp_path):
    rows = [
        _covered("LAG"),
        _txn("LAG", BUY_FIELD, 301, "2021-02-01", "2021-02-03"),
        _txn("LAG", BUY_FIELD, 302, "2021-02-10", "2021-02-12"),
        # transacted before as_of but FILED after -> must be invisible on 03-31
        _txn("LAG", BUY_FIELD, 303, "2021-03-25", "2021-04-05"),
    ]
    store = _store(tmp_path, rows)

    on_mar31 = insider_cluster_score("LAG", date(2021, 3, 31), store=store)
    assert on_mar31.components["distinct_buyers"] == 2  # 303 not yet filed
    assert on_mar31.cluster_detected is False

    after_filing = insider_cluster_score("LAG", date(2021, 4, 6), store=store)
    assert after_filing.components["distinct_buyers"] == 3  # 303 now visible
    assert after_filing.cluster_detected is True

    assert_no_future_rows(store, [BUY_FIELD, COVERAGE_FIELD], ["LAG"],
                          [date(2021, 3, 31), date(2021, 4, 6), date(2021, 5, 1)])
