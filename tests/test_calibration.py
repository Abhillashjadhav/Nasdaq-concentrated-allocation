"""Calibration eval runner (ARCHITECTURE.md §8, §9.11).

Runs the stats-engine calibration: it must recover a known planted lift, and it
must NOT manufacture an edge on a zero-lift set. The eval logic lives in
``evals.calibration``; here we execute it under pytest across a few seeds so the
recovery is not a single-seed fluke.
"""

from __future__ import annotations

import pytest

from evals.calibration import (
    assert_recovers_planted_lift,
    assert_zero_lift_not_significant,
)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_recovers_planted_lift(seed):
    res = assert_recovers_planted_lift(seed=seed)
    # both arms are reported (two-arm, never winners-only)
    assert res.p_fired > res.base_rate > res.p_not_fired


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_zero_lift_not_significant(seed):
    assert_zero_lift_not_significant(seed=seed)
