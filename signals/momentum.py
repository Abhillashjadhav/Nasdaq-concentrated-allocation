"""Price momentum signal (ARCHITECTURE.md §4, §9.5).

A locked core signal. ``momentum_score(ticker, as_of)`` reads the close series
ONLY through ``store.get_data`` (no peeking) and combines four classic momentum
components into a 0–100 sub-score:

1. **12-1 month return** — trailing 12-month price change *excluding the most
   recent month* (`price[t-21] / price[t-252] - 1`), which avoids the well-known
   short-term reversal effect.
2. **Price vs 200-DMA** — `price / mean(last 200 closes) - 1`.
3. **200-DMA slope** — `sma200(now) / sma200(21d ago) - 1`; its sign says whether
   the 200-DMA is rising or falling.
4. **52-week-high proximity** — `price / max(last 252 closes) - 1` (<= 0); 0 means
   the price is at its 52-week high.

Lookbacks are in TRADING DAYS (row offsets on the point-in-time series), so the
computation is fully deterministic and the golden case is decimal-exact.

Score mapping (documented, per §8)
----------------------------------
Each raw component is mapped to 0–100 by a bounded-linear band: a value at the
band's midpoint scores 50, the top of the band 100, the bottom 0, with clamping
outside. The four sub-scores are combined
``0.4·mom + 0.25·trend + 0.15·slope + 0.2·proximity``. This per-ticker mapping is
monotonic in momentum strength. A cross-sectional *percentile* (the §8
alternative) needs the universe cross-section and so belongs to the
stats/calibration layer, not this pure per-ticker function.

Insufficient history is NEVER scored as a silent 0 or 50: it returns an explicit
``insufficient_data`` flag + reason, surfaceable as a coverage gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

import store as store_pkg

LOOKBACK_12M = 252  # trading days ~ 12 months
LOOKBACK_1M = 21    # trading days ~ 1 month (the "-1" in 12-1)
SMA_WINDOW = 200    # trading days for the moving average
SLOPE_LOOKBACK = 21  # trading days back to measure 200-DMA slope
HIGH_52W_WINDOW = 252  # trading days ~ 52 weeks for the rolling high
MIN_HISTORY = LOOKBACK_12M + 1  # need index 252 to exist -> 253 closes

# (low, high) raw-value bands mapped to 0..100; midpoint -> 50.
MOM_BAND = (-0.5, 0.5)
TREND_BAND = (-0.2, 0.2)
SLOPE_BAND = (-0.1, 0.1)
PROX_BAND = (-0.5, 0.0)  # at the 52wk high (0) -> 100; 50% below -> 0
WEIGHTS = {
    "mom_12_1": 0.4,
    "price_vs_sma200": 0.25,
    "sma200_slope": 0.15,
    "high_52w_proximity": 0.2,
}


@dataclass
class MomentumScore:
    ticker: str
    as_of: date
    score: float | None  # 0..100, or None when insufficient data
    insufficient_data: bool
    reason: str | None = None
    components: dict = dc_field(default_factory=dict)


def _band(x: float, lo: float, hi: float) -> float:
    """Map raw value ``x`` onto 0..100: lo->0, midpoint->50, hi->100, clamped."""
    frac = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, frac)) * 100.0


def momentum_score(ticker: str, as_of: date, *, store=None) -> MomentumScore:
    """Compute the 0–100 momentum sub-score for ``ticker`` as of ``as_of``.

    Reads only via ``store.get_data`` (knowledge_date <= as_of). Returns a
    ``MomentumScore`` whose ``insufficient_data`` is True (with a reason) when the
    ticker lacks the required history — never a silent score.
    """
    store = store or store_pkg
    rows = store.get_data("close", ticker, as_of)  # newest event_date first
    n = len(rows)
    if n < MIN_HISTORY:
        return MomentumScore(
            ticker=ticker, as_of=as_of, score=None, insufficient_data=True,
            reason=f"insufficient_history: need {MIN_HISTORY} closes, have {n}",
        )

    prices = rows["value"].to_numpy(dtype=float)  # index 0 == most recent
    mom_12_1 = prices[LOOKBACK_1M] / prices[LOOKBACK_12M] - 1.0
    sma_now = prices[:SMA_WINDOW].mean()
    sma_prev = prices[SLOPE_LOOKBACK:SLOPE_LOOKBACK + SMA_WINDOW].mean()
    price_vs_sma200 = prices[0] / sma_now - 1.0
    sma200_slope = sma_now / sma_prev - 1.0
    high_52w_proximity = prices[0] / prices[:HIGH_52W_WINDOW].max() - 1.0

    raw = {
        "mom_12_1": mom_12_1,
        "price_vs_sma200": price_vs_sma200,
        "sma200_slope": sma200_slope,
        "high_52w_proximity": high_52w_proximity,
    }
    sub = {
        "mom_12_1": _band(mom_12_1, *MOM_BAND),
        "price_vs_sma200": _band(price_vs_sma200, *TREND_BAND),
        "sma200_slope": _band(sma200_slope, *SLOPE_BAND),
        "high_52w_proximity": _band(high_52w_proximity, *PROX_BAND),
    }
    score = sum(WEIGHTS[k] * sub[k] for k in WEIGHTS)

    return MomentumScore(
        ticker=ticker, as_of=as_of, score=score, insufficient_data=False,
        components={"raw": raw, "subscores": sub},
    )


def insufficient_history_gaps(tickers: Sequence[str], as_of: date, *, store=None) -> list[dict]:
    """Surface tickers that can't be momentum-scored as coverage gaps (same
    shape evals.coverage uses), so a thin history is reported, not silently 0/50."""
    gaps = []
    for t in tickers:
        res = momentum_score(t, as_of, store=store)
        if res.insufficient_data:
            gaps.append(
                {"ticker": t, "field": "close", "reason": "insufficient_history",
                 "vendor": "store"}
            )
    return gaps
