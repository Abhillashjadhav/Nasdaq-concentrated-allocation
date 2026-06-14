"""Earnings-estimate revision breadth (net % of analysts raising).
Reads only via ``store.get_data``.

Stub — implemented in PR 6 with its golden case (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date


def compute(ticker: str, as_of: date):
    raise NotImplementedError("Estimate-revision breadth signal lands in PR 6")
