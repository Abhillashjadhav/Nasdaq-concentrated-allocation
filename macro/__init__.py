"""Macro regime gate (ARCHITECTURE.md §4, §9.9).

Risk-off -> cash when HY OAS widening AND Fed tightening hard AND VIX regime
breaking. An AND gate: all conditions must coincide. Reads only via
``store.get_data``; fails safe to risk-off when macro inputs are missing.

Implementation lives in ``macro.regime``.
"""

from __future__ import annotations

from .regime import RegimeState, macro_coverage_gaps, regime_state

__all__ = ["regime_state", "RegimeState", "macro_coverage_gaps"]
