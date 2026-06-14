"""Walk-forward tests (ARCHITECTURE.md §2, §3, §9.12).

Proves the discipline: a consistent edge shows up across most slices; a one-year
fluke is correctly judged NOT consistent; purge/embargo removes boundary-crossing
observations (effective N shrinks); a zero-lift set stays ~0 across slices.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.walk_forward import purge_embargo, run_walk_forward
from evals.calibration import synthetic_observations

FIRE = {"threshold": 80.0, "n_boot": 400, "seed": 0}  # forwarded to two_arm_lift


def _year(year, *, fired_rate, not_rate, ic=0.0, balanced=False, n=600, seed=0):
    obs = synthetic_observations(
        n=n, fired_win_rate=fired_rate, not_fired_win_rate=not_rate,
        ic_strength=ic, balanced=balanced, seed=seed,
    )
    obs["as_of"] = pd.Timestamp(f"{year}-01-01")
    obs["ticker"] = obs["ticker"].astype(str) + f"_{year}"
    return obs


def test_consistent_lift_holds_across_slices():
    obs = pd.concat(
        [_year(y, fired_rate=0.70, not_rate=0.45, ic=0.02, seed=y) for y in range(2016, 2023)],
        ignore_index=True,
    )
    res = run_walk_forward(obs, **FIRE)

    assert res.effective_n_slices == 7
    assert res.frac_slices_positive == 1.0
    assert res.n_years_significant_positive >= 3
    assert res.consistent is True
    assert res.mean_lift > 0.10
    # two-arm reported every slice (never winners-only)
    assert all(s.p_fired > s.base_rate > s.p_not_fired for s in res.slices)


def test_one_year_fluke_is_not_consistent():
    years = []
    for y in range(2016, 2023):
        if y == 2018:
            years.append(_year(y, fired_rate=0.70, not_rate=0.45, ic=0.02, seed=y))
        else:
            years.append(_year(y, fired_rate=0.50, not_rate=0.50, balanced=True, seed=y))
    res = run_walk_forward(pd.concat(years, ignore_index=True), **FIRE)

    assert res.n_years_significant_positive == 1
    assert res.consistent is False  # one good year is not >=3 years -> fluke caught


def test_purge_embargo_removes_boundary_crossers():
    jan = _year(2018, fired_rate=0.70, not_rate=0.45, n=50, seed=1)  # window ends on boundary
    jul = synthetic_observations(n=50, seed=2)
    jul["as_of"] = pd.Timestamp("2018-07-01")  # window crosses 2019-01-01
    jul["ticker"] = jul["ticker"].astype(str) + "_jul"

    df = pd.concat([jan, jul], ignore_index=True)
    kept, n_removed = purge_embargo(df, horizon_months=12, embargo_days=21)

    assert n_removed == 50                       # every mid-year row purged
    assert len(kept) == 50                       # only the Jan-1 entries survive
    assert (pd.to_datetime(kept["as_of"]) == pd.Timestamp("2018-01-01")).all()


def test_zero_lift_stays_flat_across_slices():
    obs = pd.concat(
        [_year(y, fired_rate=0.50, not_rate=0.50, balanced=True, seed=y) for y in range(2016, 2023)],
        ignore_index=True,
    )
    res = run_walk_forward(obs, **FIRE)

    assert res.n_years_significant_positive == 0
    assert res.consistent is False
    assert abs(res.mean_lift) < 0.02
