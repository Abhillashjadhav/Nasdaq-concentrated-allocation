"""Survivorship-free universe construction + liquidity filter.

The universe at any historical date MUST include companies later delisted,
merged, or bankrupt (ARCHITECTURE.md §2). Using today's survivors inflates
returns 1–4%/yr and can reverse conclusions. Scope: Nasdaq Composite, healthcare
+ technology, all cap tiers, then a liquidity filter.

Implementation lives in ``universe.universe`` and reads only through
``store.get_data`` (ARCHITECTURE.md §7, §9.4).
"""

from __future__ import annotations

from .universe import (
    NullSectorClassifier,
    SectorClassifier,
    UniverseResult,
    build_universe,
    get_universe,
)

__all__ = [
    "get_universe",
    "build_universe",
    "UniverseResult",
    "SectorClassifier",
    "NullSectorClassifier",
]
