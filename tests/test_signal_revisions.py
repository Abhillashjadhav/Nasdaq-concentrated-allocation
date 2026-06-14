"""Estimate-revision-breadth tests (ARCHITECTURE.md §4, §8, §9.6).

Golden case: two snapshots with a known up/down split -> a decimal-exact breadth
sub-score, derived independently of the implementation. Plus insufficient-history,
low-analyst-coverage, and no-future-leak guards.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from signals.revisions import FIELDS, revision_breadth_score
from store.store import PITStore

ALL_FIELDS = list(FIELDS.values())


def _snapshot(ticker, kd, consensus, up_cum, down_cum, n_analysts) -> pd.DataFrame:
    vals = {
        FIELDS["consensus"]: consensus, FIELDS["up_cum"]: up_cum,
        FIELDS["down_cum"]: down_cum, FIELDS["n_analysts"]: n_analysts,
    }
    kd = pd.Timestamp(kd)
    return pd.DataFrame(
        [{"ticker": ticker, "field": f, "value": float(v),
          "event_date": kd, "knowledge_date": kd, "source": "fixture"}
         for f, v in vals.items()]
    )


def test_golden_case(tmp_path):
    store = PITStore(tmp_path / "rev.sqlite")
    store.put_data(
        pd.concat(
            [
                _snapshot("REVUP", "2020-01-31", 2.00, 10, 5, 10),
                _snapshot("REVUP", "2020-02-28", 2.10, 17, 8, 10),
            ],
            ignore_index=True,
        )
    )
    res = revision_breadth_score("REVUP", date(2020, 3, 15), store=store)

    # up_window = 17-10 = 7, down_window = 8-5 = 3 -> breadth = (7-3)/10 = 0.4
    # band(-1,1): (0.4+1)/2 * 100 = 70.0
    assert res.insufficient_data is False
    assert res.score == pytest.approx(70.0, abs=1e-9)
    assert res.components["raw"]["breadth"] == pytest.approx(0.4, abs=1e-12)
    assert res.components["raw"]["up_window"] == pytest.approx(7.0, abs=1e-12)
    assert res.components["raw"]["down_window"] == pytest.approx(3.0, abs=1e-12)
    assert res.components["raw"]["consensus_delta"] == pytest.approx(0.10, abs=1e-9)


def test_net_down_scores_below_50(tmp_path):
    store = PITStore(tmp_path / "down.sqlite")
    store.put_data(
        pd.concat(
            [
                _snapshot("CUTS", "2020-01-31", 3.00, 4, 4, 10),
                _snapshot("CUTS", "2020-02-28", 2.50, 5, 11, 10),  # 1 up, 7 down
            ],
            ignore_index=True,
        )
    )
    res = revision_breadth_score("CUTS", date(2020, 3, 15), store=store)
    # breadth = ((5-4) - (11-4)) / 10 = (1 - 7)/10 = -0.6 -> band = 20.0
    assert res.score == pytest.approx(20.0, abs=1e-9)


def test_insufficient_history(tmp_path):
    store = PITStore(tmp_path / "one.sqlite")
    store.put_data(_snapshot("SOLO", "2020-01-31", 2.00, 10, 5, 10))
    res = revision_breadth_score("SOLO", date(2020, 2, 15), store=store)
    assert res.insufficient_data is True
    assert res.score is None
    assert "insufficient_revision_history" in res.reason


def test_low_analyst_coverage_flagged_not_scored(tmp_path):
    store = PITStore(tmp_path / "thin.sqlite")
    store.put_data(
        pd.concat(
            [
                _snapshot("THIN", "2020-01-31", 2.00, 1, 0, 2),
                _snapshot("THIN", "2020-02-28", 2.05, 2, 0, 2),  # only 2 analysts
            ],
            ignore_index=True,
        )
    )
    res = revision_breadth_score("THIN", date(2020, 3, 15), store=store)
    assert res.insufficient_data is True
    assert res.low_coverage is True
    assert res.score is None
    assert "low_analyst_coverage" in res.reason


def test_no_future_leak(tmp_path):
    """A score as of T must not depend on snapshots known after T."""
    snaps = [
        _snapshot("REVUP", "2020-01-31", 2.00, 10, 5, 10),
        _snapshot("REVUP", "2020-02-28", 2.10, 17, 8, 10),
        _snapshot("REVUP", "2020-04-30", 5.00, 99, 8, 10),  # future, must not leak
    ]
    full = PITStore(tmp_path / "full.sqlite")
    full.put_data(pd.concat(snaps, ignore_index=True))
    partial = PITStore(tmp_path / "partial.sqlite")
    partial.put_data(pd.concat(snaps[:2], ignore_index=True))

    as_of = date(2020, 3, 15)  # after the 2nd snapshot, before the future one
    assert revision_breadth_score("REVUP", as_of, store=full).score == pytest.approx(
        revision_breadth_score("REVUP", as_of, store=partial).score, abs=1e-12
    )
    assert_no_future_rows(full, ALL_FIELDS, ["REVUP"],
                          [date(2020, 2, 1), as_of, date(2020, 5, 1)])
