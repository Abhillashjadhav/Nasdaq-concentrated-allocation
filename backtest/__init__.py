"""Backtest: winner labeling + walk-forward runner.

Entry Jan 1 each year 2016–2026; 12-month forward horizon. The winner labeler
(beat Nasdaq Composite total return by >=15pp over the next 12 months) lives in
``backtest.labels``; the purge/embargo walk-forward lives in
``backtest.walk_forward``.

The winner labeler is the ONE sanctioned look-ahead (ARCHITECTURE.md §3) — it is
quarantined here and never called from the signal path.
"""

from __future__ import annotations

from .labels import BENCHMARK_TICKER, WinnerLabel, label_winner
from .walk_forward import (
    SliceResult,
    WalkForwardResult,
    purge_embargo,
    run_walk_forward,
)

__all__ = [
    "label_winner", "WinnerLabel", "BENCHMARK_TICKER",
    "run_walk_forward", "WalkForwardResult", "SliceResult", "purge_embargo",
]
