"""Two-arm stats engine unit tests (ARCHITECTURE.md §2, §8, §9.11).

Covers the schema range asserts, the drop-unknown-labels rule, a hand-checkable
two-arm computation, and the fail-loud guards (empty arm / no labels).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from stats.two_arm import OBSERVATION_SCHEMA, two_arm_lift

AS_OF = pd.Timestamp("2018-01-01")


def _obs(score, is_winner, excess=None):
    return {"ticker": "T", "as_of": AS_OF, "score": float(score),
            "is_winner": is_winner, "excess_return": score if excess is None else excess}


def _frame(rows):
    return pd.DataFrame(rows)


def test_hand_checkable_two_arm(tmp_path):
    rows = (
        [_obs(90, True) for _ in range(4)] + [_obs(90, False)]      # fired: 4/5 win
        + [_obs(10, True)] + [_obs(10, False) for _ in range(4)]    # not-fired: 1/5 win
        + [_obs(90, None), _obs(10, None)]                          # unknown -> dropped
    )
    res = two_arm_lift(_frame(rows), threshold=50.0)
    assert res.n == 10 and res.n_dropped_unknown == 2
    assert res.n_fired == 5 and res.n_not_fired == 5
    assert res.base_rate == pytest.approx(0.5)
    assert res.p_fired == pytest.approx(0.8)
    assert res.p_not_fired == pytest.approx(0.2)        # loser arm explicitly counted
    assert res.lift == pytest.approx(0.3)               # 0.8 - 0.5
    assert res.rank_ic > 0                              # excess == score -> monotonic


def test_unknown_labels_never_counted():
    # all-unknown -> no answer key -> fail loud, not a silent 0
    rows = [_obs(90, None), _obs(10, None)]
    with pytest.raises(ValueError):
        two_arm_lift(_frame(rows), threshold=50.0)


def test_score_out_of_range_rejected():
    rows = [_obs(150, True), _obs(10, False)]  # 150 > 100
    with pytest.raises(SchemaErrors):
        two_arm_lift(_frame(rows), threshold=50.0)


def test_is_winner_invalid_value_rejected():
    bad = _frame([{"ticker": "T", "as_of": AS_OF, "score": 90.0,
                   "is_winner": "maybe", "excess_return": 1.0}])
    with pytest.raises(SchemaErrors):
        OBSERVATION_SCHEMA.validate(bad, lazy=True)


def test_empty_arm_fails_loud():
    rows = [_obs(90, True), _obs(90, False)]  # threshold 100 -> nothing fires
    with pytest.raises(ValueError):
        two_arm_lift(_frame(rows), threshold=100.0)
