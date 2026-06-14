"""Two-arm statistics engine (ARCHITECTURE.md §2, §3, §8).

The experiment: for each signal, P(winner | signal fired) vs the unconditional
base rate, with same-signal losers explicitly counted. Computes conditional
win-rate lift, a bootstrap CI, a p-value, and rank-IC. A winners-only view is
forbidden — the result always carries the base rate and both arms.

Implementation lives in ``stats.two_arm``.
"""

from __future__ import annotations

from .two_arm import OBSERVATION_SCHEMA, TwoArmResult, two_arm_lift

__all__ = ["two_arm_lift", "TwoArmResult", "OBSERVATION_SCHEMA"]
