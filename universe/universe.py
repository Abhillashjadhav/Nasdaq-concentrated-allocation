"""Survivorship-free universe + liquidity filter (ARCHITECTURE.md §7, §9.4).

``get_universe(as_of)`` returns the Nasdaq healthcare + technology tickers that
were *tradable as of that date* — built entirely from ``store.get_data`` so no
module reads raw data and no future row can leak in (§2).

Design notes (why the shape is what it is)
-------------------------------------------
* Enumeration: the chokepoint is per-ticker, so the caller supplies a
  survivorship-free CANDIDATE symbol set (every name that ever existed, incl.
  delisted). We never default to today's survivors — the point-in-time
  membership adapter that would build that set is a follow-up.
* Survivorship: a name is tradable as of ``as_of`` iff it has a ``close`` within
  ``staleness_days`` of ``as_of``. A name delisted in 2010 is therefore in the
  2008 universe and out of the 2020 one, with no "exists today" filter.
* Liquidity: price floor (latest close) + average-dollar-volume floor (trailing
  mean of ``dollar_volume``), both point-in-time. ``dollar_volume`` ingestion is
  a follow-up; missing coverage is SURFACED, never silently dropped.
* Sector: behind a ``SectorClassifier`` interface. The default returns ``None``
  for every name (the float-only record schema can't hold a categorical sector,
  so a real sector source is a follow-up) — sector is never faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence

import pandas as pd

import store as store_pkg
from evals.coverage import coverage_report

PRICE_FLOOR = 5.0
ADV_FLOOR = 1_000_000.0
ADV_WINDOW = 63  # ~one quarter of trading days
STALENESS_DAYS = 30  # no close within this window => not tradable as of the date
ALLOWED_SECTORS = {"healthcare", "technology"}


class SectorClassifier(Protocol):
    """Maps a ticker to its sector as of a date, or ``None`` if unknown."""

    def classify(self, ticker: str, as_of: date) -> str | None: ...


class NullSectorClassifier:
    """Default placeholder: never fakes a sector (always ``None``).

    A real classifier needs a categorical sector source (e.g. the Sharadar
    tickers table), which the float-only point-in-time record schema can't hold —
    so wiring it is a follow-up. Passing this (or ``None``) disables the sector
    filter and the result flags ``sector_filter_applied=False``.
    """

    def classify(self, ticker: str, as_of: date) -> str | None:
        return None


@dataclass
class UniverseResult:
    as_of: date
    tickers: list[str]
    excluded: list[dict]  # {ticker, reason} — every drop is recorded, never silent
    sector_filter_applied: bool

    @property
    def coverage_gaps(self) -> list[dict]:
        """Exclusions caused by missing data (vs. a real filter decision)."""
        gaps = {"no_price_coverage", "adv_coverage_gap", "sector_unknown"}
        return [e for e in self.excluded if e["reason"] in gaps]


def build_universe(
    as_of: date,
    candidates: Sequence[str],
    *,
    price_floor: float = PRICE_FLOOR,
    adv_floor: float = ADV_FLOOR,
    adv_window: int = ADV_WINDOW,
    staleness_days: int = STALENESS_DAYS,
    sector_classifier: SectorClassifier | None = None,
    store=None,
) -> UniverseResult:
    """Filter a survivorship-free candidate set down to the tradable, liquid,
    in-sector universe as of ``as_of``. Returns a result that records every
    exclusion (and coverage gap) so nothing shrinks the list silently."""
    store = store or store_pkg
    sector_filter_applied = sector_classifier is not None
    candidates = list(dict.fromkeys(candidates))  # de-dupe, preserve order
    tickers: list[str] = []
    excluded: list[dict] = []

    # Surface price-coverage gaps via the PR-3 coverage report (don't shrink silently).
    cov = coverage_report(store, ["close"], candidates, as_of, vendor="store")
    no_price = {t for (t, _f) in cov.quarantined}

    cutoff = pd.Timestamp(as_of).normalize()
    for t in candidates:
        if t in no_price:
            excluded.append({"ticker": t, "reason": "no_price_coverage"})
            continue

        closes = store.get_data("close", t, as_of)  # newest event_date first
        last = closes.iloc[0]
        if (cutoff - pd.Timestamp(last["event_date"]).normalize()).days > staleness_days:
            excluded.append({"ticker": t, "reason": "not_listed_as_of"})
            continue
        if float(last["value"]) < price_floor:
            excluded.append({"ticker": t, "reason": "below_price_floor"})
            continue

        dv = store.get_data("dollar_volume", t, as_of)
        if dv.empty:
            excluded.append({"ticker": t, "reason": "adv_coverage_gap"})
            continue
        if float(dv["value"].head(adv_window).mean()) < adv_floor:
            excluded.append({"ticker": t, "reason": "below_adv_floor"})
            continue

        if sector_filter_applied:
            sector = sector_classifier.classify(t, as_of)
            if sector is None:
                excluded.append({"ticker": t, "reason": "sector_unknown"})
                continue
            if sector.lower() not in ALLOWED_SECTORS:
                excluded.append({"ticker": t, "reason": "sector_excluded"})
                continue

        tickers.append(t)

    return UniverseResult(
        as_of=as_of,
        tickers=tickers,
        excluded=excluded,
        sector_filter_applied=sector_filter_applied,
    )


def get_universe(as_of: date, candidates: Sequence[str] | None = None, **kwargs) -> list[str]:
    """Return the tradable Nasdaq healthcare+tech universe as of ``as_of``.

    ``candidates`` is the survivorship-free symbol set to consider; it is
    required because defaulting to today's listed names would reintroduce
    survivorship bias (§2). The membership adapter that builds it is a follow-up.
    """
    if candidates is None:
        raise ValueError(
            "get_universe requires a survivorship-free candidate symbol set "
            "(incl. delisted names); the point-in-time membership adapter is a "
            "follow-up. Refusing to default to today's survivors (survivorship bias)."
        )
    return build_universe(as_of, candidates, **kwargs).tickers
