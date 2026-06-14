"""Two-arm statistics engine (ARCHITECTURE.md §2, §3, §8).

This is the experiment. For a signal, given observations across (ticker × as_of),
it computes P(winner | signal fired) AND the unconditional base rate P(winner),
the conditional lift, a bootstrap CI on the lift, a p-value vs the null of no
lift, and rank-IC.

The TWO-ARM non-negotiable (§2.3) is enforced structurally
----------------------------------------------------------
The only result this module produces — ``TwoArmResult`` — always carries
``base_rate`` and BOTH arms (``p_fired`` and ``p_not_fired``, the same-signal
losers, explicitly counted). There is no function that returns a fired-arm
win-count in isolation. A signal "looks good" only if it beats the base rate, not
by having a high absolute number of wins. An empty arm fails loud.

Incomplete outcomes are dropped, never counted
----------------------------------------------
Observations whose label is ``not_yet_known`` (``is_winner is None``) have no
answer key, so they are excluded entirely — never scored as a win or a loss.
Observations without a score (signal couldn't be computed) are likewise excluded.

"Fired" definition
------------------
``fired = score >= threshold``. By default the threshold is the top-quantile cut
(``fire_quantile=0.8`` → the top 20% of scores). It is documented and overridable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera.pandas as pa
from scipy.stats import norm, spearmanr

# IO-level schema (§8) with the deferred range asserts.
OBSERVATION_SCHEMA = pa.DataFrameSchema(
    {
        "ticker": pa.Column(str, nullable=False),
        "as_of": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "score": pa.Column(float, pa.Check.in_range(0.0, 100.0), nullable=True, coerce=True),
        # dtype unpinned: a labels column is bool when complete, object when it
        # carries None (not_yet_known). We only constrain the VALUES.
        "is_winner": pa.Column(
            nullable=True,
            checks=pa.Check(lambda s: s.isin([True, False]) | s.isna(),
                            error="is_winner must be True, False, or None"),
        ),
        "excess_return": pa.Column(float, nullable=True, coerce=True),
    },
    strict=True,
    ordered=False,
)


@dataclass
class TwoArmResult:
    signal: str | None
    n: int                # labeled, scored observations used
    n_fired: int
    n_not_fired: int
    base_rate: float      # P(winner) unconditional
    p_fired: float        # P(winner | fired)
    p_not_fired: float    # P(winner | not fired) — the loser arm, counted
    lift: float           # p_fired - base_rate (§3 headline)
    lift_vs_not_fired: float
    ci_low: float
    ci_high: float
    p_value: float        # two-proportion test, fired vs not-fired
    rank_ic: float        # Spearman(score, excess_return)
    threshold: float
    n_dropped_unknown: int


def two_arm_lift(
    observations: pd.DataFrame,
    *,
    signal: str | None = None,
    fire_quantile: float = 0.8,
    threshold: float | None = None,
    ci: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> TwoArmResult:
    """Compute the two-arm comparison for a signal. See module docstring."""
    df = OBSERVATION_SCHEMA.validate(observations, lazy=True)

    labeled = df[df["is_winner"].notna() & df["score"].notna()].copy()
    n_dropped = int(len(df) - len(labeled))
    if labeled.empty:
        raise ValueError("no labeled, scored observations to evaluate")

    score = labeled["score"].to_numpy(dtype=float)
    win = labeled["is_winner"].astype(bool).to_numpy().astype(float)

    cut = threshold if threshold is not None else float(np.quantile(score, fire_quantile))
    fired = score >= cut
    n_fired, n_not = int(fired.sum()), int((~fired).sum())
    if n_fired == 0 or n_not == 0:
        raise ValueError(
            f"fire threshold {cut:.4g} yields an empty arm "
            f"(fired={n_fired}, not_fired={n_not}); cannot run a two-arm test"
        )

    base_rate = float(win.mean())
    p_fired = float(win[fired].mean())
    p_not_fired = float(win[~fired].mean())
    lift = p_fired - base_rate

    ci_low, ci_high = _bootstrap_lift_ci(win, fired, ci=ci, n_boot=n_boot, seed=seed)
    p_value = _two_proportion_pvalue(win[fired].sum(), n_fired, win[~fired].sum(), n_not)
    rank = _rank_ic(labeled)

    return TwoArmResult(
        signal=signal, n=int(len(labeled)), n_fired=n_fired, n_not_fired=n_not,
        base_rate=base_rate, p_fired=p_fired, p_not_fired=p_not_fired,
        lift=lift, lift_vs_not_fired=p_fired - p_not_fired,
        ci_low=ci_low, ci_high=ci_high, p_value=p_value, rank_ic=rank,
        threshold=cut, n_dropped_unknown=n_dropped,
    )


def _bootstrap_lift_ci(win, fired, *, ci, n_boot, seed):
    """Percentile bootstrap CI on lift = P(win|fired) - base_rate, at a FIXED
    threshold (resample observations, recompute the fixed rule's lift)."""
    rng = np.random.default_rng(seed)
    n = len(win)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w, f = win[idx], fired[idx]
        if f.sum() == 0:
            continue
        boots.append(w[f].mean() - w.mean())
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _two_proportion_pvalue(x1, n1, x2, n2):
    """Two-sided two-proportion z-test, fired vs not-fired (null: equal rates)."""
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return float(2.0 * norm.sf(abs(z)))


def _rank_ic(labeled: pd.DataFrame) -> float:
    """Spearman rank correlation of score vs forward excess return."""
    sub = labeled[labeled["excess_return"].notna()]
    if len(sub) < 3 or sub["score"].nunique() < 2 or sub["excess_return"].nunique() < 2:
        return float("nan")
    rho, _ = spearmanr(sub["score"].to_numpy(), sub["excess_return"].to_numpy())
    return float(rho)
