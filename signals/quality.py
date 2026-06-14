"""Quality / profitability (Piotroski F-score, gross profitability).
Used as a failure filter (ARCHITECTURE.md §4). Reads only via ``store.get_data``.

Stub — implemented in PR 8 with its golden case (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def compute(ticker: str, as_of: date):
    raise NotImplementedError("Quality/profitability signal lands in PR 8")
