"""Winner labeler — the answer key (ARCHITECTURE.md §3, §9.10).

``label_winner(ticker, as_of)`` returns the binary outcome: did the ticker's total
return over the 12 months AFTER ``as_of`` beat the Nasdaq benchmark's return over
the same window by at least 15 percentage points.

THE ONE SANCTIONED LOOK-AHEAD (read this carefully)
---------------------------------------------------
A label, by definition, uses the future — that is legitimate for an answer key.
But it is quarantined to the labeler, and conflating it with the signal path is
the subtlest possible bug in the whole harness. The rules:

* Signals NEVER import or call this module. They read ``store.get_data(..., as_of)``
  (decision time). The labeler reads ``store.get_data(..., horizon_end)`` — i.e.
  it queries the store AS OF the horizon end (``as_of`` + 12 months), which is the
  only place a later ``as_of`` is used on purpose.
* A label computed for ``as_of`` is used ONLY to grade a prediction that was made
  at ``as_of``. It is never fed back into a signal, a universe, or a gate.

So the no-peek chokepoint is not bypassed — the labeler simply asks the store what
was known by GRADING time, which is a different (later) decision date than the one
the prediction was made on.

Incomplete future (never label prematurely)
-------------------------------------------
If the 12-month window has not fully elapsed in the data, the outcome is not yet
known and we return a flag rather than a partial/zero label. Completeness is
judged on the BENCHMARK (the index always trades): if the benchmark has no close
within ``tolerance_days`` of ``horizon_end``, the future hasn't happened yet.

Delisting = disaster, not "unknown"
-----------------------------------
If the benchmark window HAS elapsed but the ticker stopped trading mid-window, the
name delisted. We model that conservatively as a total loss (exit value 0 ->
-100% return) — the disaster case the survivorship-free harness must recognize —
not as a missing/unknown outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date

import pandas as pd

import store as store_pkg

PRICE_FIELD = "close"
BENCHMARK_TICKER = "^IXIC"  # Nasdaq Composite (yfinance symbol; Stooq code mapped in data/prices.py)
HORIZON_MONTHS = 12
WIN_THRESHOLD_PP = 15.0  # beat the benchmark by >= 15 percentage points
TOLERANCE_DAYS = 10      # how close to horizon_end a close must be for the window to count


@dataclass
class WinnerLabel:
    ticker: str
    as_of: date
    horizon_end: date
    is_winner: bool | None  # None when the outcome is not yet known
    outcome_known: bool
    reason: str | None = None
    components: dict = dc_field(default_factory=dict)


def _price_at(store, ticker, when) -> tuple[float | None, pd.Timestamp | None]:
    """Latest close on/before ``when`` (the sanctioned look-ahead point), with its
    event_date. ``None`` when no close exists."""
    rows = store.get_data(PRICE_FIELD, ticker, when)
    if rows.empty:
        return None, None
    latest = rows.iloc[0]
    return float(latest["value"]), pd.Timestamp(latest["event_date"]).normalize()


def label_winner(
    ticker: str,
    as_of: date,
    *,
    store=None,
    benchmark: str = BENCHMARK_TICKER,
    horizon_months: int = HORIZON_MONTHS,
    threshold_pp: float = WIN_THRESHOLD_PP,
    tolerance_days: int = TOLERANCE_DAYS,
) -> WinnerLabel:
    """Grade ``ticker`` over the 12 months after ``as_of`` against the benchmark.
    Returns a binary winner label, or an outcome-not-yet-known flag when the future
    window has not fully elapsed. THE SANCTIONED LOOK-AHEAD — never call from a
    signal."""
    store = store or store_pkg
    as_of_ts = pd.Timestamp(as_of).normalize()
    horizon_end = (as_of_ts + pd.DateOffset(months=horizon_months)).normalize()
    horizon_floor = horizon_end - pd.Timedelta(days=tolerance_days)

    def _flag(reason):
        return WinnerLabel(ticker, as_of, horizon_end.date(), None, False, reason=reason)

    # Window completeness is judged on the benchmark (the index always trades).
    bench_exit, bench_exit_date = _price_at(store, benchmark, horizon_end)
    if bench_exit is None or bench_exit_date < horizon_floor:
        return _flag(
            f"outcome_not_yet_known: benchmark {benchmark!r} has no close within "
            f"{tolerance_days}d of {horizon_end.date()}"
        )
    bench_entry, _ = _price_at(store, benchmark, as_of)
    if bench_entry is None:
        return _flag(f"benchmark_unavailable_at_entry: {benchmark!r} @ {as_of}")

    entry_price, _ = _price_at(store, ticker, as_of)
    if entry_price is None:
        return _flag(f"no_entry_price: {ticker!r} @ {as_of}")

    # Benchmark window elapsed, so a ticker with no recent close delisted mid-window.
    exit_price, exit_date = _price_at(store, ticker, horizon_end)
    delisted = exit_price is None or exit_date < horizon_floor
    effective_exit = 0.0 if delisted else exit_price  # delisting -> total loss

    ticker_return = effective_exit / entry_price - 1.0
    benchmark_return = bench_exit / bench_entry - 1.0
    excess_return = ticker_return - benchmark_return
    is_winner = excess_return >= threshold_pp / 100.0  # >= : the boundary is inclusive

    return WinnerLabel(
        ticker, as_of, horizon_end.date(), is_winner, True,
        components={
            "ticker_return": ticker_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "entry_price": entry_price,
            "exit_price": effective_exit,
            "delisted": delisted,
        },
    )
