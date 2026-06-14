"""Macro regime gate (ARCHITECTURE.md §4, §9.9).

``regime_state(as_of)`` combines the locked macro conditions into a single
risk-off determination, reading ONLY through ``store.get_data``. It returns a
regime STATE and a boolean GATE flag — not a 0–100 score: the gate either lets
the book stay invested (open) or forces it to cash (closed).

The three conditions (all measured point-in-time over a trailing window)
----------------------------------------------------------------------------
* **HY OAS widening** — high-yield option-adjusted spread rose by at least
  ``oas_widening_pp`` over the window (credit stress building).
* **Fed tightening (hard)** — the policy rate rose by at least ``rate_rise_pp``
  over the window (direction = tightening).
* **VIX regime breaking** — VIX is at/above ``vix_level`` OR spiked by at least
  ``vix_spike_pp`` over the window.

The gate is an **AND**: ``risk_off = widening AND tightening AND vix_break``. A
single dangerous condition is NOT enough — they must coincide. (Documented so the
AND, not OR, is unambiguous.)

Point-in-time correctness
-------------------------
Each macro series is a PIT record stored under the sentinel ticker ``MACRO`` with
``event_date`` = the data date and ``knowledge_date`` = the release date (FRED has
small publication lags). The store filters ``knowledge_date <= as_of``, so a data
point not yet released is invisible.

Safe default on missing data
----------------------------
If any required series is missing/too short as of the date, we DO NOT assume
risk-on — staying invested while blind to macro stress is the dangerous default.
We **fail safe to risk-off** (gate closed → cash), set ``insufficient_data``, and
name the missing series so it is never a silent assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

import pandas as pd

import store as store_pkg

MACRO_TICKER = "MACRO"  # sentinel ticker for non-ticker-specific macro series
HY_OAS = "hy_oas"
POLICY_RATE = "fed_funds_rate"
VIX = "vix"

WINDOW_DAYS = 63          # ~one quarter of trading days
OAS_WIDENING_PP = 1.0     # +1.0pp (100bps) widening over the window
RATE_RISE_PP = 0.5        # policy rate up >= 0.5pp over the window (tightening hard)
VIX_LEVEL = 30.0          # VIX at/above this level
VIX_SPIKE_PP = 10.0       # or VIX up >= 10 points over the window


@dataclass
class RegimeState:
    as_of: date
    state: str            # "risk_off" | "risk_on"
    gate_closed: bool     # True => go to cash (defensive)
    insufficient_data: bool
    conditions: dict = dc_field(default_factory=dict)   # the three booleans
    components: dict = dc_field(default_factory=dict)    # raw values
    missing: list = dc_field(default_factory=list)
    reason: str | None = None


def _current_and_lookback(rows, window_start):
    """From newest-first rows, return (current value, lookback value). Lookback is
    the most recent observation on/before ``window_start`` (or None if none)."""
    current = float(rows.iloc[0]["value"])
    older = rows[pd.to_datetime(rows["event_date"]).dt.normalize() <= window_start]
    lookback = float(older.iloc[0]["value"]) if not older.empty else None
    return current, lookback


def regime_state(
    as_of: date,
    *,
    store=None,
    window_days: int = WINDOW_DAYS,
    oas_widening_pp: float = OAS_WIDENING_PP,
    rate_rise_pp: float = RATE_RISE_PP,
    vix_level: float = VIX_LEVEL,
    vix_spike_pp: float = VIX_SPIKE_PP,
) -> RegimeState:
    """Determine the macro regime as of ``as_of``. Reads only via store.get_data;
    fails safe to risk-off (gate closed) when inputs are missing."""
    store = store or store_pkg
    cutoff = pd.Timestamp(as_of).normalize()
    window_start = cutoff - pd.Timedelta(days=window_days)

    oas = store.get_data(HY_OAS, MACRO_TICKER, as_of)
    rate = store.get_data(POLICY_RATE, MACRO_TICKER, as_of)
    vix = store.get_data(VIX, MACRO_TICKER, as_of)

    # Resolve current/lookback; OAS and rate are change-based and need a lookback.
    missing: list[str] = []
    oas_now = oas_lb = rate_now = rate_lb = vix_now = vix_lb = None
    if oas.empty:
        missing.append(f"{HY_OAS}:absent")
    else:
        oas_now, oas_lb = _current_and_lookback(oas, window_start)
        if oas_lb is None:
            missing.append(f"{HY_OAS}:insufficient_history")
    if rate.empty:
        missing.append(f"{POLICY_RATE}:absent")
    else:
        rate_now, rate_lb = _current_and_lookback(rate, window_start)
        if rate_lb is None:
            missing.append(f"{POLICY_RATE}:insufficient_history")
    if vix.empty:
        missing.append(f"{VIX}:absent")
    else:
        vix_now, vix_lb = _current_and_lookback(vix, window_start)  # lookback optional

    if missing:
        return RegimeState(
            as_of, "risk_off", gate_closed=True, insufficient_data=True,
            missing=missing,
            reason="macro inputs missing -> fail-safe to risk-off (gate closed)",
            components={"hy_oas": oas_now, "fed_funds_rate": rate_now, "vix": vix_now},
        )

    hy_widening = (oas_now - oas_lb) >= oas_widening_pp
    fed_tightening = (rate_now - rate_lb) >= rate_rise_pp
    vix_breaking = (vix_now >= vix_level) or (
        vix_lb is not None and (vix_now - vix_lb) >= vix_spike_pp
    )
    risk_off = hy_widening and fed_tightening and vix_breaking

    return RegimeState(
        as_of,
        state="risk_off" if risk_off else "risk_on",
        gate_closed=risk_off,
        insufficient_data=False,
        conditions={
            "hy_widening": hy_widening,
            "fed_tightening": fed_tightening,
            "vix_breaking": vix_breaking,
        },
        components={
            "hy_oas": oas_now, "hy_oas_change": oas_now - oas_lb,
            "fed_funds_rate": rate_now, "rate_change": rate_now - rate_lb,
            "vix": vix_now, "vix_change": None if vix_lb is None else vix_now - vix_lb,
        },
    )


def macro_coverage_gaps(as_of: date, *, store=None) -> list[dict]:
    """Surface missing macro series as coverage gaps (same shape evals.coverage
    uses), so the fail-safe-to-risk-off is visible, not silent."""
    res = regime_state(as_of, store=store)
    return [
        {"ticker": MACRO_TICKER, "field": m.split(":")[0], "reason": m.split(":")[1],
         "vendor": "store"}
        for m in res.missing
    ]
