"""Quality / profitability signal (ARCHITECTURE.md §4, §9.8).

A locked core signal, used as a failure filter. ``quality_score(ticker, as_of)``
computes the Piotroski F-score (9 binary tests) plus gross profitability, reading
ONLY through ``store.get_data``.

Point-in-time correctness (the critical part)
---------------------------------------------
Fundamentals are public only on/after the FILING date, never the period-end date,
and the lag is large (a Q4 ending Dec 31 may not file until late Feb/Mar). So each
line item is stored as a PIT record with ``event_date`` = period end and
``knowledge_date`` = filing date. The store filters ``knowledge_date <= as_of``,
so a period that has ended but not yet filed is correctly invisible — this is the
easiest place to accidentally peek, and the no-leak test proves we don't.

Fundamental fields (one PIT record per fiscal period each)
----------------------------------------------------------
net_income, cfo (operating cash flow), total_assets, long_term_debt,
current_assets, current_liabilities, shares_outstanding, gross_profit, revenue.
The latest two FILED periods (by period end) are used; the "rising/falling" tests
need two periods, so a ticker with fewer is flagged, not scored.

F-score (0..9): NI>0; CFO>0; ROA rising; CFO>NI; long-term-debt ratio falling;
current ratio rising; no new shares; gross margin rising; asset turnover rising.
Ratios use period-end total assets (not averages) for a deterministic, exactly
reproducible score.

Score mapping (documented, per §8)
----------------------------------
``0.7 * (F/9 * 100) + 0.3 * band(gross_profitability, 0, 0.5)``. Gross
profitability = gross_profit / total_assets. A cross-sectional percentile
(preferred by §8) is deferred to the stats/calibration layer, consistent with the
other signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

import store as store_pkg

MIN_PERIODS = 2  # the rising/falling checks need a current and a prior period
GP_BAND = (0.0, 0.5)  # gross profitability -> 0..100; 0.25 -> 50
WEIGHTS = {"f_score": 0.7, "gross_profitability": 0.3}
ANCHOR = "total_assets"
FIELDS = [
    "net_income", "cfo", "total_assets", "long_term_debt", "current_assets",
    "current_liabilities", "shares_outstanding", "gross_profit", "revenue",
]
# denominators that must be strictly positive in both periods
_DENOMS = ["total_assets", "revenue", "current_liabilities"]


@dataclass
class QualityScore:
    ticker: str
    as_of: date
    score: float | None  # 0..100, or None when not scoreable
    f_score: int | None
    insufficient_data: bool
    reason: str | None = None
    components: dict = dc_field(default_factory=dict)


def _band(x: float, lo: float, hi: float) -> float:
    """Map raw value ``x`` onto 0..100: lo->0, midpoint->50, hi->100, clamped."""
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) * 100.0


def _flag(ticker, as_of, reason) -> "QualityScore":
    return QualityScore(ticker, as_of, None, None, True, reason=reason)


def quality_score(ticker: str, as_of: date, *, store=None) -> QualityScore:
    """Compute the quality sub-score for ``ticker`` as of ``as_of`` from the two
    most recent FILED fiscal periods. Reads only via store.get_data."""
    store = store or store_pkg

    series = {f: store.get_data(f, ticker, as_of) for f in FIELDS}  # newest period first
    if series[ANCHOR].empty:
        return _flag(ticker, as_of, "no_fundamental_coverage")
    if min(len(s) for s in series.values()) < MIN_PERIODS:
        return _flag(ticker, as_of, "insufficient_fundamental_history")

    t = {f: float(series[f].iloc[0]["value"]) for f in FIELDS}  # latest period
    p = {f: float(series[f].iloc[1]["value"]) for f in FIELDS}  # prior period
    if any(t[d] <= 0 or p[d] <= 0 for d in _DENOMS):
        return _flag(ticker, as_of, "invalid_fundamentals")

    roa_t, roa_p = t["net_income"] / t["total_assets"], p["net_income"] / p["total_assets"]
    checks = {
        "ni_positive": t["net_income"] > 0,
        "cfo_positive": t["cfo"] > 0,
        "roa_rising": roa_t > roa_p,
        "accrual_cfo_gt_ni": t["cfo"] > t["net_income"],
        "ltd_ratio_falling":
            t["long_term_debt"] / t["total_assets"] < p["long_term_debt"] / p["total_assets"],
        "current_ratio_rising":
            t["current_assets"] / t["current_liabilities"]
            > p["current_assets"] / p["current_liabilities"],
        "no_new_shares": t["shares_outstanding"] <= p["shares_outstanding"],
        "gross_margin_rising":
            t["gross_profit"] / t["revenue"] > p["gross_profit"] / p["revenue"],
        "asset_turnover_rising":
            t["revenue"] / t["total_assets"] > p["revenue"] / p["total_assets"],
    }
    f_score = sum(1 for passed in checks.values() if passed)  # 0..9
    gross_profitability = t["gross_profit"] / t["total_assets"]

    f_component = f_score / 9.0 * 100.0
    gp_component = _band(gross_profitability, *GP_BAND)
    score = WEIGHTS["f_score"] * f_component + WEIGHTS["gross_profitability"] * gp_component

    return QualityScore(
        ticker, as_of, score, f_score, False,
        components={
            "checks": checks,
            "gross_profitability": gross_profitability,
            "f_component": f_component,
            "gp_component": gp_component,
        },
    )


def quality_coverage_gaps(tickers: Sequence[str], as_of: date, *, store=None) -> list[dict]:
    """Surface tickers that can't be quality-scored as coverage gaps (same shape
    evals.coverage uses) — flagged, not scored."""
    gaps = []
    for ticker in tickers:
        res = quality_score(ticker, as_of, store=store)
        if res.insufficient_data:
            gaps.append({"ticker": ticker, "field": ANCHOR, "reason": res.reason,
                         "vendor": "store"})
    return gaps
