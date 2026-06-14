"""Boosters / tie-breakers (less proven): revenue acceleration (growth rate
rising, in dollars) and Rule of 40 (software). Reads only via ``store.get_data``.

Stub — implemented after the core signals (ARCHITECTURE.md §4, §9).
"""

from __future__ import annotations

from datetime import date


def revenue_acceleration(ticker: str, as_of: date):
    raise NotImplementedError("Revenue-acceleration booster lands with the boosters PR")


def rule_of_40(ticker: str, as_of: date):
    raise NotImplementedError("Rule-of-40 booster lands with the boosters PR")
