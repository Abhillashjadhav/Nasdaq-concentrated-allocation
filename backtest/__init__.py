"""Walk-forward backtest runner with purge/embargo.

Entry Jan 1 each year 2016–2026; 12-month forward horizon. The winner labeler
(beat Nasdaq Composite total return by >=15pp over the next 12 months) and the
purge/embargo walk-forward live here.

Winner labeler: PR 10. Walk-forward + purge/embargo: PR 12 (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def label_winner(ticker: str, entry: date):
    """True if ``ticker`` beat the Nasdaq Composite total return by >=15pp over
    the 12 months after ``entry``."""
    raise NotImplementedError("Winner labeler lands in PR 10")


def run_walk_forward(*args, **kwargs):
    """Walk-forward across entry years with purge/embargo to prevent leakage."""
    raise NotImplementedError("Walk-forward + purge/embargo runner lands in PR 12")
