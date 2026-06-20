"""Nasdaq healthcare + technology universe builder (SimFin sector classification).

Classifies each candidate symbol into technology / healthcare using SimFin's
company + industry reference data — the SAME vendor that feeds the quality signal,
so the universe and the fundamentals come from one source with no per-ticker SEC
throttle. The classification is CACHED into the point-in-time store as a ``sector``
record so it runs once, not every run; membership is then resolved point-in-time.

Point-in-time membership (ARCHITECTURE.md §6): the cached record's
``knowledge_date`` is the company's earliest known SimFin Report Date (when its
financials first became public), so for a past as-of date a name is included only
if it was already reporting then.

SURVIVORSHIP CAVEAT (§2.2): the candidate set comes from today's listed symbols
(free Nasdaq Trader files lack delisted names — the $0 gap). So the historical
universe here is SURVIVOR-LIMITED. We say so explicitly in the report and never
pretend it is survivorship-free.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from datetime import date

import pandas as pd

from data.simfin_client import DEFAULT_CACHE_DIR, load_universe_reference
from store.schema import COLUMNS

log = logging.getLogger(__name__)
SECTOR_FIELD = "sector"
SOURCE = "simfin"
_FAR_FUTURE = date(2100, 1, 1)  # "is this symbol classified at all?" probe

TECHNOLOGY = "technology"
HEALTHCARE = "healthcare"
# Stored as a small float code (the universal record's value is a float); decoded
# back to a label on read. Only in-scope names are cached, so the universe read is
# a simple membership lookup.
_CODES = {TECHNOLOGY: 1.0, HEALTHCARE: 2.0}
_DECODE = {v: k for k, v in _CODES.items()}

# SimFin "Sector" labels (matched case-insensitively) that map into our investable
# universe. SimFin's reference sectors are coarse strings; we accept the common
# spellings for each so a vendor relabel ("Health Care" vs "Healthcare") still maps.
_TECH_SECTORS = {"technology", "information technology"}
_HEALTH_SECTORS = {"healthcare", "health care"}


def classify_sector(sector) -> str | None:
    """Return ``"technology"`` / ``"healthcare"`` for an in-scope SimFin sector
    label, else ``None`` (the name is excluded from the universe). Absent/blank -> None."""
    if sector is None or (isinstance(sector, float) and pd.isna(sector)):
        return None
    s = str(sector).strip().lower()
    if s in _TECH_SECTORS:
        return TECHNOLOGY
    if s in _HEALTH_SECTORS:
        return HEALTHCARE
    return None


@dataclass
class ClassifyResult:
    n_symbols: int = 0
    n_classified: int = 0      # had a SimFin sector (in OR out of scope)
    n_kept: int = 0            # of those, in-scope (technology/healthcare) -> cached
    n_cached: int = 0          # already in the store (skipped)
    n_quarantined: int = 0     # absent from SimFin / no report date -> no PIT record
    quarantine: list[dict] = dc_field(default_factory=list)


def _coerce_report_date(value):
    """Earliest Report Date -> Timestamp, or None when missing/unparseable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _cache_one(store, sym, label, first_report) -> None:
    """Persist ONE ticker's sector to the store immediately (incremental cache).
    knowledge_date = earliest Report Date, so membership is point-in-time."""
    kd = pd.Timestamp(first_report)
    store.put_data(pd.DataFrame([{
        "ticker": sym, "field": SECTOR_FIELD, "value": _CODES[label],
        "event_date": kd, "knowledge_date": kd, "source": SOURCE,
    }], columns=COLUMNS))


def classify_and_cache(symbols, *, store, reference_loader=None, api_key=None,
                       cache_dir=DEFAULT_CACHE_DIR, refresh: bool = False,
                       limit: int | None = None, log_every: int = 10) -> ClassifyResult:
    """Classify each symbol's sector from SimFin reference data and cache the
    in-scope (tech/healthcare) names in the store INCREMENTALLY. On a re-run an
    already-cached ticker is skipped (no work) unless ``refresh`` is set. A name
    absent from SimFin — or with no Report Date to anchor a point-in-time
    knowledge_date — is quarantined (counted, never fatal). ``limit`` caps the
    symbols processed (smoke tests). ``reference_loader`` is injected in tests to
    bypass the network; it returns a ``[Ticker, Sector, first_report]`` frame."""
    syms = list(symbols)
    if limit is not None:
        syms = syms[:limit]
    log_every = max(1, log_every)
    total = len(syms)
    res = ClassifyResult(n_symbols=total)

    loader = reference_loader or load_universe_reference
    ref = loader(api_key=api_key, cache_dir=cache_dir, refresh=refresh, tickers=syms)
    lut: dict[str, tuple] = {}
    for _, r in ref.iterrows():
        lut[str(r["Ticker"]).upper()] = (r.get("Sector"), r.get("first_report"))

    n_cached_start = 0 if refresh else sum(
        not store.get_data(SECTOR_FIELD, s, _FAR_FUTURE).empty for s in syms)
    print(f"building universe: {total} symbols to classify "
          f"(cached: {n_cached_start}, to fetch: {total - n_cached_start})", flush=True)

    t0 = time.perf_counter()
    for i, sym in enumerate(syms, 1):
        if not refresh and not store.get_data(SECTOR_FIELD, sym, _FAR_FUTURE).empty:
            res.n_cached += 1
        else:
            sector, first_report = lut.get(sym.upper(), (None, None))
            first = _coerce_report_date(first_report)
            if sector is None or (isinstance(sector, float) and pd.isna(sector)) or first is None:
                # No SimFin sector, or no Report Date to anchor a PIT knowledge_date.
                res.n_quarantined += 1
                res.quarantine.append({"ticker": sym, "field": SECTOR_FIELD,
                                       "reason": "absent_from_simfin", "vendor": SOURCE})
            else:
                res.n_classified += 1
                label = classify_sector(sector)
                if label is not None:
                    _cache_one(store, sym, label, first)  # only in-scope names cached
                    res.n_kept += 1
                # out-of-scope sector (e.g. Financials) -> classified but excluded
        if i % log_every == 0 or i == total:
            log.info("classified %d/%d (kept %d hc+tech, skipped %d, failed %d)",
                     i, total, res.n_kept, res.n_cached, res.n_quarantined)
            print(f"classified {i}/{total} (kept {res.n_kept}, skipped {res.n_cached}, "
                  f"failed {res.n_quarantined}, cache-hits {res.n_cached})", flush=True)
    print(f"universe build complete: {res.n_kept} names in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return res


def nasdaq_hc_tech_universe(as_of, symbols, *, store) -> list[str]:
    """The tech+healthcare members reporting as-of ``as_of`` (read from the cached
    sector record, point-in-time via get_data). Survivor-limited — see module docstring."""
    members = []
    for sym in symbols:
        rows = store.get_data(SECTOR_FIELD, sym, as_of)  # only if knowledge_date <= as_of
        if rows.empty:
            continue
        if _DECODE.get(float(rows.iloc[0]["value"])) in (TECHNOLOGY, HEALTHCARE):
            members.append(sym)
    return members
