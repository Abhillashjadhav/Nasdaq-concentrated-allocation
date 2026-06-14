"""Calibration eval for the two-arm stats engine (ARCHITECTURE.md §8).

The validity test: inject a synthetic signal+label set with a KNOWN planted lift
and assert the engine recovers it. Plus a non-vacuity check — a zero-lift set must
report ~0 lift with a non-significant p-value, proving the engine does not
manufacture an edge that isn't there.

The planted construction: the top ``fire_frac`` of observations score in [80,100]
(the fired arm) and win at ``fired_win_rate``; the rest score in [0,80) and win at
``not_fired_win_rate``. With fired=0.70, not-fired=0.45, fire_frac=0.20 the base
rate is 0.50 and the true lift over base is 0.20. ``excess_return`` is score plus
noise scaled by ``ic_strength`` so rank-IC is positive when a real edge exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stats.two_arm import two_arm_lift

FIRE_THRESHOLD = 80.0  # fired arm scores live in [80, 100]


def synthetic_observations(
    *,
    n: int = 6000,
    fire_frac: float = 0.20,
    fired_win_rate: float = 0.70,
    not_fired_win_rate: float = 0.45,
    ic_strength: float = 0.02,
    noise: float = 1.0,
    balanced: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic observation frame with a known planted lift.

    ``balanced=True`` assigns wins deterministically so each arm hits EXACTLY its
    target rate (no sampling noise) — used for the zero-lift non-vacuity check, so
    its non-significance is robust rather than a flaky draw from the null.
    """
    rng = np.random.default_rng(seed)
    n_fired = int(round(n * fire_frac))
    n_not = n - n_fired

    score = np.concatenate([rng.uniform(80, 100, n_fired), rng.uniform(0, 80, n_not)])
    if balanced:
        win = np.concatenate([
            np.arange(n_fired) < round(n_fired * fired_win_rate),
            np.arange(n_not) < round(n_not * not_fired_win_rate),
        ])
    else:
        win = np.concatenate([
            rng.random(n_fired) < fired_win_rate,
            rng.random(n_not) < not_fired_win_rate,
        ])
    excess = ic_strength * score + rng.normal(0, noise, n)
    as_of = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")

    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "as_of": as_of,
        "score": score,
        "is_winner": win.astype(object),
        "excess_return": excess,
    })


def assert_recovers_planted_lift(seed: int = 0):
    """A planted 0.20 lift (0.70 fired vs 0.50 base) must be recovered within the
    CI, the CI must exclude zero, and rank-IC must be positive."""
    obs = synthetic_observations(seed=seed)  # defaults plant lift = 0.20
    res = two_arm_lift(obs, signal="planted", threshold=FIRE_THRESHOLD, n_boot=1000, seed=seed)

    assert 0.15 <= res.lift <= 0.25, f"lift {res.lift} not near planted 0.20"
    assert res.ci_low <= 0.20 <= res.ci_high, "CI must contain the planted lift"
    assert res.ci_low > 0.0, "CI must exclude zero (a real edge)"
    assert res.p_value < 0.01, f"edge must be significant (p={res.p_value})"
    assert res.rank_ic > 0.0, f"rank-IC must be positive (got {res.rank_ic})"
    return res


def assert_zero_lift_not_significant(seed: int = 0):
    """Non-vacuity: a zero-lift set (fired wins at the base rate) must report ~0
    lift, a CI straddling zero, and a non-significant p-value."""
    obs = synthetic_observations(
        fired_win_rate=0.50, not_fired_win_rate=0.50, ic_strength=0.0,
        balanced=True, seed=seed,
    )
    res = two_arm_lift(obs, signal="null", threshold=FIRE_THRESHOLD, n_boot=1000, seed=seed)

    assert abs(res.lift) < 0.05, f"zero-lift set produced lift {res.lift}"
    assert res.ci_low <= 0.0 <= res.ci_high, "CI must straddle zero"
    assert res.p_value > 0.05, f"must NOT be significant (p={res.p_value})"
    return res
