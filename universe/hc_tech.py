"""Nasdaq healthcare + technology universe builder (classify once, cache, rank).

Resolves each candidate symbol to its SEC SIC code via the EDGAR submissions API
(CIK from the existing ``CikResolver``), keeps only technology + healthcare names
(``universe.sic``), and CACHES the classification into the point-in-time store as
a ``sic`` record so classification runs once, not every run. Tickers that fail
CIK / SIC / submissions resolution are QUARANTINED (counted, never fatal).

Point-in-time membership (ARCHITECTURE.md §6): the ``sic`` record's
``knowledge_date`` is the company's earliest known filing date, so for a past
as-of date a name is included only if it was already filing then.

SURVIVORSHIP CAVEAT (§2.2): the candidate set comes from today's listed symbols
(free Nasdaq Trader files lack delisted names — the $0 gap). So the historical
universe here is SURVIVOR-LIMITED. We say so explicitly in the report and never
pretend it is survivorship-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from datetime import date

import pandas as pd

from data.edgar_client import EdgarHTTPError, UnknownTickerError
from store.schema import COLUMNS
from universe.sic import classify_sic

log = logging.getLogger(__name__)
SIC_FIELD = "sic"
SOURCE = "edgar"
_FAR_FUTURE = date(2100, 1, 1)  # "is this symbol classified at all?" probe


@dataclass
class ClassifyResult:
    n_symbols: int = 0
    n_classified: int = 0      # newly cached this run
    n_cached: int = 0          # already in the store (skipped)
    n_quarantined: int = 0
    quarantine: list[dict] = dc_field(default_factory=list)


def _earliest_filing_date(submissions: dict):
    fdates = submissions.get("filings", {}).get("recent", {}).get("filingDate", [])
    return min(fdates) if fdates else None


def classify_and_cache(symbols, *, client, resolver, store) -> ClassifyResult:
    """Resolve+classify each symbol's SIC and cache it in the store (idempotent:
    symbols already cached are skipped). Failures are quarantined, never fatal."""
    res = ClassifyResult(n_symbols=len(symbols))
    rows = []
    for sym in symbols:
        if not store.get_data(SIC_FIELD, sym, _FAR_FUTURE).empty:
            res.n_cached += 1
            continue
        try:
            cik = resolver.resolve(sym)
            submissions = client.get_json(f"/submissions/CIK{cik}.json")
            sic = submissions.get("sic") or submissions.get("sicCode")
            first_filing = _earliest_filing_date(submissions)
            if not sic or first_filing is None:
                res.n_quarantined += 1
                res.quarantine.append({"ticker": sym, "field": SIC_FIELD,
                                       "reason": "no_sic_or_filing", "vendor": SOURCE})
                continue
            kd = pd.Timestamp(first_filing)
            rows.append({"ticker": sym, "field": SIC_FIELD, "value": float(int(sic)),
                         "event_date": kd, "knowledge_date": kd, "source": SOURCE})
            res.n_classified += 1
        except (UnknownTickerError, EdgarHTTPError) as exc:
            res.n_quarantined += 1
            res.quarantine.append({"ticker": sym, "field": SIC_FIELD,
                                   "reason": f"sic_unavailable: {exc}", "vendor": SOURCE})
    if rows:
        store.put_data(pd.DataFrame(rows, columns=COLUMNS))
    log.info("universe classify: %d symbols, %d newly classified, %d cached, %d quarantined",
             res.n_symbols, res.n_classified, res.n_cached, res.n_quarantined)
    return res


def nasdaq_hc_tech_universe(as_of, symbols, *, store) -> list[str]:
    """The tech+healthcare members filing as-of ``as_of`` (read from cached SIC,
    point-in-time via get_data). Survivor-limited — see module docstring."""
    members = []
    for sym in symbols:
        rows = store.get_data(SIC_FIELD, sym, as_of)  # only if filing knowledge_date <= as_of
        if rows.empty:
            continue
        if classify_sic(rows.iloc[0]["value"]) in ("technology", "healthcare"):
            members.append(sym)
    return members
