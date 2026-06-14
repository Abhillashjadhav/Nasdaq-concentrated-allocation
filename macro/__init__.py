"""Macro regime gate (ARCHITECTURE.md §4).

Risk-off -> cash when HY OAS widening AND Fed tightening hard AND VIX regime
breaking. An AND gate: all conditions must fire. Reads only via
``store.get_data``.

Implemented in PR 9 with its eval (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def regime(as_of: date) -> str:
    """Return the macro regime as of ``as_of`` (e.g. ``"risk_on"`` /
    ``"risk_off"``). Risk-off forces the book to cash."""
    raise NotImplementedError("Macro gate + regime classification land in PR 9")
