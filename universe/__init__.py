"""Survivorship-free universe construction + liquidity filter.

The universe at any historical date MUST include companies later delisted,
merged, or bankrupt (ARCHITECTURE.md §2). Using today's survivors inflates
returns 1–4%/yr and can reverse conclusions. Scope: Nasdaq Composite, healthcare
+ technology, all cap tiers, then a liquidity filter.

Implemented in PR 4 (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def get_universe(as_of: date) -> list[str]:
    """Return the liquidity-filtered, survivorship-free ticker universe as it
    stood on ``as_of`` — including names later delisted."""
    raise NotImplementedError("Universe + liquidity filter land in PR 4")
