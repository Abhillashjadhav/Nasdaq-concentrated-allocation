"""Walk-forward evaluation with purge + embargo (ARCHITECTURE.md §2, §9.12).

``run_walk_forward`` slices the observation frame by entry year, runs the §9.11
two-arm engine PER slice, and reports lift / CI / rank-IC per year plus an
aggregate. The deliverable is the DISTRIBUTION of the edge over time — does it
hold across >=3 distinct years (§3, §10) — not a single pooled number.

No leakage across time (the §2 non-negotiable, applied to time)
--------------------------------------------------------------
Labels look 12 months forward, so an observation's outcome can leak into an
adjacent slice. Two defenses, applied before any slicing:

* PURGE — drop an observation whose label window ``(as_of, as_of + horizon)``
  STRICTLY straddles a year boundary: its outcome spans two slices, so keeping it
  would leak test-period information backward. With annual Jan-1 entries the
  windows are adjacent (a Jan-1 window ends exactly on the next boundary), so
  clean annual entries survive while intra-year observations are purged.
* EMBARGO — additionally drop an observation whose ``as_of`` falls within
  ``embargo_days`` AFTER a boundary, or whose window END falls within
  ``embargo_days`` BEFORE a boundary, killing residual autocorrelation right
  around the boundary. (Entries exactly on a boundary are not embargoed.)

Overlap-aware counting (don't inflate significance)
---------------------------------------------------
Because 12-month labels overlap, pooling every observation into one significance
test would double-count. We do NOT pool: each YEAR is the independent temporal
unit. Per-slice significance is reported, and the headline is >=3-year
consistency. Raw vs kept observation counts are reported so the de-overlapping is
explicit; within a slice, contemporaneous (same-window) names are cross-sectional
and counted once each.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from stats.two_arm import OBSERVATION_SCHEMA, two_arm_lift

CONSISTENCY_YEARS = 3  # an edge must hold in >=3 distinct years (§3, §10)


@dataclass
class SliceResult:
    year: int
    n: int
    base_rate: float
    p_fired: float
    p_not_fired: float
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    rank_ic: float

    @property
    def significant_positive(self) -> bool:
        return self.ci_low > 0.0  # CI excludes zero on the positive side


@dataclass
class WalkForwardResult:
    slices: list[SliceResult] = dc_field(default_factory=list)
    n_observations_raw: int = 0
    n_observations_kept: int = 0
    n_purged: int = 0
    skipped_years: list[int] = dc_field(default_factory=list)

    @property
    def effective_n_slices(self) -> int:
        return len(self.slices)  # independent temporal units, not pooled rows

    @property
    def mean_lift(self) -> float:
        return float(np.mean([s.lift for s in self.slices])) if self.slices else float("nan")

    @property
    def median_lift(self) -> float:
        return float(np.median([s.lift for s in self.slices])) if self.slices else float("nan")

    @property
    def frac_slices_positive(self) -> float:
        return _frac(s.lift > 0 for s in self.slices)

    @property
    def frac_slices_significant(self) -> float:
        return _frac(s.ci_low > 0 or s.ci_high < 0 for s in self.slices)

    @property
    def n_years_significant_positive(self) -> int:
        return sum(s.significant_positive for s in self.slices)

    @property
    def consistent(self) -> bool:
        return self.n_years_significant_positive >= CONSISTENCY_YEARS

    @property
    def mean_rank_ic(self) -> float:
        ics = [s.rank_ic for s in self.slices if not np.isnan(s.rank_ic)]
        return float(np.mean(ics)) if ics else float("nan")


def _frac(bools) -> float:
    vals = list(bools)
    return float(np.mean(vals)) if vals else float("nan")


def purge_embargo(
    observations: pd.DataFrame, *, horizon_months: int = 12, embargo_days: int = 21
) -> tuple[pd.DataFrame, int]:
    """Remove observations whose 12-month label window crosses (purge) or sits
    within ``embargo_days`` of (embargo) a year boundary. Returns (kept, n_removed)."""
    a = pd.to_datetime(observations["as_of"]).dt.normalize()
    w = a + pd.DateOffset(months=horizon_months)
    emb = pd.Timedelta(days=embargo_days)
    years = range(int(a.dt.year.min()), int(a.dt.year.max()) + 2)
    boundaries = [pd.Timestamp(y, 1, 1) for y in years]

    remove = pd.Series(False, index=observations.index)
    for b in boundaries:
        crosses = (a < b) & (w > b)
        near = ((a > b) & (a <= b + emb)) | ((w >= b - emb) & (w < b))
        remove |= crosses | near
    kept = observations[~remove]
    return kept, int(remove.sum())


def run_walk_forward(
    observations: pd.DataFrame,
    *,
    horizon_months: int = 12,
    embargo_days: int = 21,
    min_obs_per_slice: int = 20,
    **two_arm_kwargs,
) -> WalkForwardResult:
    """Run the two-arm engine per entry-year slice after purge/embargo. See module
    docstring. Raises if no slice is evaluable (fail loud, never a silent empty)."""
    df = OBSERVATION_SCHEMA.validate(observations, lazy=True)
    kept, n_purged = purge_embargo(df, horizon_months=horizon_months, embargo_days=embargo_days)

    result = WalkForwardResult(
        n_observations_raw=int(len(df)),
        n_observations_kept=int(len(kept)),
        n_purged=n_purged,
    )
    kept = kept.assign(_year=pd.to_datetime(kept["as_of"]).dt.year)
    for year, group in kept.groupby("_year"):
        if len(group) < min_obs_per_slice:
            result.skipped_years.append(int(year))
            continue
        try:
            r = two_arm_lift(group.drop(columns="_year"), signal=str(year), **two_arm_kwargs)
        except ValueError:
            result.skipped_years.append(int(year))  # e.g. an empty arm in this slice
            continue
        result.slices.append(SliceResult(
            year=int(year), n=r.n, base_rate=r.base_rate, p_fired=r.p_fired,
            p_not_fired=r.p_not_fired, lift=r.lift, ci_low=r.ci_low, ci_high=r.ci_high,
            p_value=r.p_value, rank_ic=r.rank_ic,
        ))

    if not result.slices:
        raise ValueError("no evaluable slices after purge/embargo and min-obs filtering")
    return result
