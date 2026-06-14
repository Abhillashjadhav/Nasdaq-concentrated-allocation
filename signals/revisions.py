"""Earnings-estimate revision-breadth signal (ARCHITECTURE.md §4, §9.6).

A locked core signal: "net % of analysts raising". ``revision_breadth_score``
reads ONLY through ``store.get_data`` (no peeking) and scores how broadly
analysts are revising FY1 EPS estimates up vs. down.

Estimate snapshot model (why it's shaped this way)
--------------------------------------------------
The universal record is ``field/value`` (a float) with no analyst dimension, so
we cannot store one row per analyst. Instead each vendor SNAPSHOT, taken at a
``knowledge_date``, is stored as four point-in-time records for the ticker:

* ``est_fy1_consensus``  — consensus FY1 EPS at the snapshot
* ``est_fy1_up_cum``     — cumulative count of UP revisions since coverage start
* ``est_fy1_down_cum``   — cumulative count of DOWN revisions since coverage start
* ``est_fy1_n_analysts`` — number of covering analysts at the snapshot

Snapshot-diff mechanism (the reusable cadence pattern)
------------------------------------------------------
Breadth is a change over time, so it is computed by DIFFERENCING the two most
recent snapshots as of ``as_of`` (this is exactly what the live 15-day cadence
will do as it accumulates snapshots)::

    up_window   = up_cum[t]   - up_cum[t-1]
    down_window = down_cum[t] - down_cum[t-1]
    breadth     = (up_window - down_window) / n_analysts[t]      # in [-1, 1]

The consensus value is differenced the same way (``consensus_delta``) and exposed
as a diagnostic. We DO NOT backfill history that doesn't exist: fewer than two
snapshots as of ``as_of`` returns an explicit ``insufficient_revision_history``
flag; an analyst count below the floor returns ``low_analyst_coverage``. Neither
is silently scored.

Score mapping (documented, per §8)
----------------------------------
``breadth`` in [-1, 1] maps to 0..100 by a bounded-linear band (-1 -> 0, 0 -> 50,
+1 -> 100). As decided for momentum, a cross-sectional percentile (preferred by
§8) needs the universe cross-section and so belongs to the stats/calibration
layer, not this pure per-ticker function.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

import store as store_pkg

MIN_SNAPSHOTS = 2  # breadth is a diff -> need a self-built history of >=2 snapshots
DEFAULT_MIN_ANALYSTS = 3  # below this, coverage is too thin to trust the breadth
BREADTH_BAND = (-1.0, 1.0)  # net fraction raising -> 0..100; 0 -> 50

FIELDS = {
    "consensus": "est_fy1_consensus",
    "up_cum": "est_fy1_up_cum",
    "down_cum": "est_fy1_down_cum",
    "n_analysts": "est_fy1_n_analysts",
}


@dataclass
class RevisionBreadthScore:
    ticker: str
    as_of: date
    score: float | None  # 0..100, or None when not scoreable
    insufficient_data: bool
    reason: str | None = None
    low_coverage: bool = False
    components: dict = dc_field(default_factory=dict)


def _band(x: float, lo: float, hi: float) -> float:
    """Map raw value ``x`` onto 0..100: lo->0, midpoint->50, hi->100, clamped."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) * 100.0


def revision_breadth_score(
    ticker: str,
    as_of: date,
    *,
    store=None,
    min_analysts: int = DEFAULT_MIN_ANALYSTS,
) -> RevisionBreadthScore:
    """Score FY1 estimate-revision breadth for ``ticker`` as of ``as_of`` by
    differencing the two most recent snapshots. Reads only via store.get_data."""
    store = store or store_pkg

    up = store.get_data(FIELDS["up_cum"], ticker, as_of)  # newest snapshot first
    down = store.get_data(FIELDS["down_cum"], ticker, as_of)
    n_an = store.get_data(FIELDS["n_analysts"], ticker, as_of)
    n_snap = min(len(up), len(down), len(n_an))
    if n_snap < MIN_SNAPSHOTS:
        return RevisionBreadthScore(
            ticker, as_of, None, True,
            reason=f"insufficient_revision_history: need {MIN_SNAPSHOTS} "
                   f"snapshots, have {n_snap}",
        )

    analysts = float(n_an.iloc[0]["value"])
    if analysts < min_analysts:
        return RevisionBreadthScore(
            ticker, as_of, None, True,
            reason=f"low_analyst_coverage: {analysts:.0f} < {min_analysts}",
            low_coverage=True,
        )

    up_window = float(up.iloc[0]["value"]) - float(up.iloc[1]["value"])
    down_window = float(down.iloc[0]["value"]) - float(down.iloc[1]["value"])
    breadth = (up_window - down_window) / analysts
    score = _band(breadth, *BREADTH_BAND)

    cons = store.get_data(FIELDS["consensus"], ticker, as_of)
    consensus_delta = (
        float(cons.iloc[0]["value"]) - float(cons.iloc[1]["value"])
        if len(cons) >= 2 else None
    )

    return RevisionBreadthScore(
        ticker, as_of, score, False,
        components={
            "raw": {
                "breadth": breadth,
                "up_window": up_window,
                "down_window": down_window,
                "n_analysts": analysts,
                "consensus_delta": consensus_delta,
            },
            "subscore": score,
        },
    )


def revision_coverage_gaps(
    tickers: Sequence[str], as_of: date, *, store=None, min_analysts: int = DEFAULT_MIN_ANALYSTS
) -> list[dict]:
    """Surface names that can't be breadth-scored as coverage gaps (same shape
    evals.coverage uses) — thin/short estimate history is reported, not scored."""
    gaps = []
    for t in tickers:
        res = revision_breadth_score(t, as_of, store=store, min_analysts=min_analysts)
        if res.insufficient_data:
            reason = "low_analyst_coverage" if res.low_coverage else "insufficient_revision_history"
            gaps.append({"ticker": t, "field": FIELDS["n_analysts"], "reason": reason,
                         "vendor": "store"})
    return gaps
