"""Backtest: winner labeling + walk-forward runner.

Entry Jan 1 each year 2016–2026; 12-month forward horizon. The winner labeler
(beat Nasdaq Composite total return by >=15pp over the next 12 months) lives in
``backtest.labels``; the purge/embargo walk-forward lands in PR 12.

The winner labeler is the ONE sanctioned look-ahead (ARCHITECTURE.md §3) — it is
quarantined here and never called from the signal path.
"""

from __future__ import annotations

from .labels import BENCHMARK_TICKER, WinnerLabel, label_winner

__all__ = ["label_winner", "WinnerLabel", "BENCHMARK_TICKER", "run_walk_forward"]


def run_walk_forward(*args, **kwargs):
    """Walk-forward across entry years with purge/embargo to prevent leakage."""
    raise NotImplementedError("Walk-forward + purge/embargo runner lands in PR 12")
