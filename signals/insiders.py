"""Insider cluster buying (>=3 opportunistic buyers, non-routine).
Reads only via ``store.get_data`` (Form 4, +2d knowledge lag).

Stub — implemented in PR 7 with its golden case (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def compute(ticker: str, as_of: date):
    raise NotImplementedError("Insider cluster-buy signal lands in PR 7")
