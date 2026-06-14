"""Report + GO/KILL verdict tests (ARCHITECTURE.md §3, §10, §9.13).

End-to-end: a consistent multi-archetype edge -> GO; a null set -> KILL; a single
confirming archetype -> MARGINAL (the bar is not fudged up to GO). Plus a
threshold test asserting the §10 cutoffs (consistency years, survivorship haircut,
sample floor, archetype count) are applied exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.walk_forward import SliceResult, WalkForwardResult, run_walk_forward
from evals.calibration import synthetic_observations
from report.build_report import build_report, evaluate_signal, resolve_verdict

FIRE = {"threshold": 80.0, "n_boot": 300, "seed": 0}


def _year(year, *, fired_rate, not_rate, ic=0.0, balanced=False, n=600, seed=0):
    obs = synthetic_observations(
        n=n, fired_win_rate=fired_rate, not_fired_win_rate=not_rate,
        ic_strength=ic, balanced=balanced, seed=seed,
    )
    obs["as_of"] = pd.Timestamp(f"{year}-01-01")
    obs["ticker"] = obs["ticker"].astype(str) + f"_{year}_{seed}"
    return obs


def _confirming_wf(tag):
    obs = pd.concat(
        [_year(y, fired_rate=0.70, not_rate=0.45, ic=0.02, seed=tag + y) for y in range(2016, 2021)],
        ignore_index=True,
    )
    return run_walk_forward(obs, **FIRE)


def _null_wf(tag):
    obs = pd.concat(
        [_year(y, fired_rate=0.50, not_rate=0.50, balanced=True, seed=tag + y) for y in range(2016, 2021)],
        ignore_index=True,
    )
    return run_walk_forward(obs, **FIRE)


def test_consistent_multi_archetype_is_go():
    rep = build_report({"momentum": _confirming_wf(0), "revisions": _confirming_wf(100)})
    assert rep.verdict == "GO"
    assert rep.n_confirmed == 2
    assert "## VERDICT: GO" in rep.markdown
    # caveats are surfaced in the rendered report
    assert any("OUT OF SCOPE" in c for c in rep.caveats)
    assert any("haircut" in c for c in rep.caveats)


def test_null_set_is_kill():
    rep = build_report({"momentum": _null_wf(0), "revisions": _null_wf(100)})
    assert rep.verdict == "KILL"
    assert rep.n_confirmed == 0


def test_single_archetype_is_marginal_not_go():
    rep = build_report({"momentum": _confirming_wf(0), "quality": _null_wf(100)})
    assert rep.verdict == "MARGINAL"      # exactly one confirms -> Partial, never GO
    assert rep.n_confirmed == 1


# --- exact §10 threshold application -------------------------------------------

def _slice(year, lift, ci_low, n_fired=200, n_not=1000, rank_ic=0.1):
    return SliceResult(
        year=year, n=n_fired + n_not, n_fired=n_fired, n_not_fired=n_not,
        base_rate=0.5, p_fired=0.5 + lift, p_not_fired=0.4, lift=lift,
        ci_low=ci_low, ci_high=ci_low + 0.3, p_value=0.01, rank_ic=rank_ic,
    )


def _wf(slices):
    return WalkForwardResult(
        slices=slices, n_observations_raw=9999, n_observations_kept=5000, n_purged=4999,
    )


def test_consistency_cutoff_is_exactly_three_years():
    three = _wf([_slice(y, 0.10, 0.05) for y in (2016, 2017, 2018)])
    two = _wf([_slice(y, 0.10, 0.05) for y in (2016, 2017)] + [_slice(2018, 0.10, -0.02)])
    assert evaluate_signal("s", three).confirmed is True
    assert evaluate_signal("s", two).confirmed is False  # 2 sig years < 3


def test_haircut_is_applied_exactly():
    # mean lift exactly equal to the 4pp haircut -> adjusted 0.0, NOT > 0 -> blocked
    at = _wf([_slice(y, 0.04, 0.01) for y in (2016, 2017, 2018)])
    above = _wf([_slice(y, 0.05, 0.01) for y in (2016, 2017, 2018)])
    v_at = evaluate_signal("s", at, survivorship_haircut_pp=4.0)
    assert v_at.confirmed is False and any("haircut" in r for r in v_at.reasons)
    assert evaluate_signal("s", above, survivorship_haircut_pp=4.0).confirmed is True


def test_sample_floor_blocks_thin_evidence():
    thin = _wf([_slice(y, 0.10, 0.05, n_fired=50, n_not=1000) for y in (2016, 2017, 2018)])
    v = evaluate_signal("s", thin, min_samples_per_arm=300)  # 150 fired < 300
    assert v.confirmed is False and any("floor" in r for r in v.reasons)


def test_rank_ic_must_be_positive():
    neg = _wf([_slice(y, 0.10, 0.05, rank_ic=-0.1) for y in (2016, 2017, 2018)])
    v = evaluate_signal("s", neg)
    assert v.confirmed is False and any("rank-IC" in r for r in v.reasons)


def test_resolve_verdict_exact_cutoffs():
    assert resolve_verdict(2) == "GO"
    assert resolve_verdict(3) == "GO"
    assert resolve_verdict(1) == "MARGINAL"
    assert resolve_verdict(0) == "KILL"


def test_empty_results_fails_loud():
    with pytest.raises(ValueError):
        build_report({})
