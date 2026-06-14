"""Price momentum signal (12-1 month relative strength; above a rising 200-DMA;
52-wk-high proximity). Reads only via ``store.get_data``.

Stub — implemented in PR 5 with its golden case (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def compute(ticker: str, as_of: date):
    raise NotImplementedError("Momentum signal lands in PR 5")
