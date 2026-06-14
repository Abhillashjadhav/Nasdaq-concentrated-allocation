"""Insider cluster-buying signal (ARCHITECTURE.md §4, §9.7).

A locked core signal. ``insider_cluster_score(ticker, as_of)`` detects a CLUSTER
of >=3 distinct insiders making genuine open-market purchases (Form 4 transaction
code ``P``) within a trailing window. It reads ONLY through ``store.get_data`` and
respects the Form 4 filing lag.

Form 4 storage model (why it's shaped this way)
-----------------------------------------------
The universal record is ``field/value`` (a float) and cannot hold both an insider
identity AND a transaction code in one row, so the CODE is encoded in the field
name and the INSIDER id in the value:

* ``form4_buy_P``  — one record per open-market purchase (code ``P``).
  ``value`` = insider id, ``event_date`` = transaction date,
  ``knowledge_date`` = filing date.
* Option exercises (``M``), grants (``A``) and sells (``S``) are routed to
  SEPARATE fields at ingest. This signal reads only ``form4_buy_P``, so non-P
  codes cannot inflate the cluster — only true opportunistic buys count.
* ``form4_covered`` — a coverage marker (value 1) written for every ticker the
  Form 4 source tracks, so we can tell "covered but no cluster" (score low) from
  "no Form 4 coverage at all" (flagged, never scored 0).

Point-in-time correctness
-------------------------
A Form 4 is public only on/after its filing date, so ``knowledge_date`` = filing
date (a few days after the trade). The store filters ``knowledge_date <= as_of``,
so a purchase that is transacted but not yet filed is correctly invisible. The
trailing window is applied to the TRANSACTION date (``event_date``).

Score mapping (documented, per §8)
----------------------------------
The count of distinct opportunistic buyers in the window maps to 0..100 via a
bounded-linear band ``(0, 6)``: 0 buyers -> 0, the >=3 cluster threshold -> 50,
6+ -> 100. A cross-sectional percentile (preferred by §8) needs the universe
cross-section and so is deferred to the stats/calibration layer, consistent with
the other signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

import pandas as pd

import store as store_pkg

MIN_CLUSTER = 3        # >=3 distinct insiders = a cluster (ARCHITECTURE.md §4)
WINDOW_DAYS = 90       # trailing window on the transaction date
CLUSTER_BAND = (0.0, 6.0)  # distinct buyers -> 0..100; the cluster threshold (3) -> 50
BUY_FIELD = "form4_buy_P"      # open-market purchases only (code P)
COVERAGE_FIELD = "form4_covered"  # marker that the ticker is tracked by the source


@dataclass
class InsiderClusterScore:
    ticker: str
    as_of: date
    score: float | None  # 0..100, or None when not scoreable
    insufficient_data: bool
    cluster_detected: bool = False
    reason: str | None = None
    components: dict = dc_field(default_factory=dict)


def _band(x: float, lo: float, hi: float) -> float:
    """Map raw value ``x`` onto 0..100: lo->0, midpoint->50, hi->100, clamped."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) * 100.0


def insider_cluster_score(
    ticker: str,
    as_of: date,
    *,
    store=None,
    window_days: int = WINDOW_DAYS,
    min_cluster: int = MIN_CLUSTER,
) -> InsiderClusterScore:
    """Score insider cluster-buying for ``ticker`` as of ``as_of``. Reads only via
    store.get_data; counts distinct insiders making code-P open-market purchases in
    the trailing window. Returns an explicit flag (not a 0) when the ticker has no
    Form 4 coverage."""
    store = store or store_pkg

    if store.get_data(COVERAGE_FIELD, ticker, as_of).empty:
        return InsiderClusterScore(
            ticker, as_of, None, True, reason="no_form4_coverage",
        )

    buys = store.get_data(BUY_FIELD, ticker, as_of)  # knowledge_date<=as_of (filed)
    cutoff = pd.Timestamp(as_of).normalize()
    window_start = cutoff - pd.Timedelta(days=window_days)
    if buys.empty:
        windowed = buys
    else:
        txn_date = pd.to_datetime(buys["event_date"]).dt.normalize()
        windowed = buys[(txn_date >= window_start) & (txn_date <= cutoff)]

    distinct_buyers = int(windowed["value"].nunique()) if not windowed.empty else 0
    score = _band(float(distinct_buyers), *CLUSTER_BAND)

    return InsiderClusterScore(
        ticker, as_of, score, False,
        cluster_detected=distinct_buyers >= min_cluster,
        components={
            "distinct_buyers": distinct_buyers,
            "n_buy_txns": int(len(windowed)),
            "window_days": window_days,
        },
    )


def insider_coverage_gaps(tickers: Sequence[str], as_of: date, *, store=None) -> list[dict]:
    """Surface tickers with no Form 4 coverage as gaps (same shape evals.coverage
    uses) — they are flagged, not scored 0."""
    gaps = []
    for t in tickers:
        res = insider_cluster_score(t, as_of, store=store)
        if res.insufficient_data:
            gaps.append({"ticker": t, "field": COVERAGE_FIELD,
                         "reason": "no_form4_coverage", "vendor": "store"})
    return gaps
