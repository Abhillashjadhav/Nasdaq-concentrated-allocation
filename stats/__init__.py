"""Two-arm statistics engine (ARCHITECTURE.md §2, §8).

The experiment: for each signal, P(winner | signal fired) vs the unconditional
base rate, with same-signal losers explicitly counted. Computes conditional
win-rate lift, confidence intervals, p-values, and rank-IC. A winners-only view
is forbidden.

Implemented in PR 11 with its calibration eval (ARCHITECTURE.md §9).
"""

from __future__ import annotations


def two_arm_lift(*args, **kwargs):
    """Return conditional win-rate lift over base rate, with the loser arm
    counted, plus CI and p-value."""
    raise NotImplementedError("Two-arm stats engine lands in PR 11")


def rank_ic(*args, **kwargs):
    """Rank information coefficient: composite score vs forward return."""
    raise NotImplementedError("Rank-IC lands in PR 11")
