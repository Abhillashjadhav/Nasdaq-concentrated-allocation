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
import threading
import time
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

# Per-ticker fetch resilience (BUG 2). A single EDGAR request can hang on an SSL
# read indefinitely — the socket timeout is not always honored — which previously
# stalled a multi-thousand-ticker build until a manual Ctrl+C. We run each fetch
# under a hard wall-clock deadline on a daemon thread (a parked thread can never
# block interpreter exit) and retry transient failures a few times; on persistent
# failure the ticker is quarantined ("failed") and the loop moves on.
_FETCH_TIMEOUT_S = 30.0
_FETCH_RETRIES = 2
_RETRY_BACKOFF_S = 1.0


class ClassificationTimeout(RuntimeError):
    """A single EDGAR classification fetch exceeded its per-ticker deadline."""


def _run_with_deadline(fn, timeout: float):
    """Run ``fn`` on a daemon thread and return its result, raising
    ``ClassificationTimeout`` if it does not finish within ``timeout`` seconds.
    A timed-out thread is abandoned (daemon -> never blocks process exit) so a
    hung SSL read cannot stall the build. Exceptions raised by ``fn`` propagate."""
    box: dict = {}

    def _worker():
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised in the caller thread below
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise ClassificationTimeout(f"fetch exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _fetch_submissions(sym, *, client, resolver, timeout, retries, sleep):
    """Resolve CIK + fetch the submissions JSON under a hard deadline, retrying
    transient timeout/HTTP errors. ``UnknownTickerError`` is permanent (the ticker
    is not in SEC's map) so it is never retried. Raises after exhausting retries,
    leaving the caller to quarantine — the result is NEVER written to the store."""
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            return _run_with_deadline(
                lambda: client.get_json(f"/submissions/CIK{resolver.resolve(sym)}.json"),
                timeout,
            )
        except UnknownTickerError:
            raise  # not in SEC's company_tickers.json -> permanent, don't retry
        except (ClassificationTimeout, EdgarHTTPError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep(_RETRY_BACKOFF_S * (attempt + 1))
    raise last_exc


@dataclass
class ClassifyResult:
    n_symbols: int = 0
    n_classified: int = 0      # newly fetched + cached this run (any SIC)
    n_kept: int = 0            # of those, in-scope (technology/healthcare)
    n_cached: int = 0          # already in the store (skipped)
    n_quarantined: int = 0
    quarantine: list[dict] = dc_field(default_factory=list)


def _earliest_filing_date(submissions: dict):
    fdates = submissions.get("filings", {}).get("recent", {}).get("filingDate", [])
    return min(fdates) if fdates else None


def _cache_one(store, sym, sic, first_filing) -> None:
    """Persist ONE ticker's SIC to the store immediately (incremental cache)."""
    kd = pd.Timestamp(first_filing)
    store.put_data(pd.DataFrame([{
        "ticker": sym, "field": SIC_FIELD, "value": float(int(sic)),
        "event_date": kd, "knowledge_date": kd, "source": SOURCE,
    }], columns=COLUMNS))


def classify_and_cache(symbols, *, client, resolver, store, refresh: bool = False,
                       limit: int | None = None, log_every: int = 10,
                       fetch_timeout: float = _FETCH_TIMEOUT_S,
                       fetch_retries: int = _FETCH_RETRIES,
                       sleep=time.sleep) -> ClassifyResult:
    """Resolve+classify each symbol's SIC and cache it in the store INCREMENTALLY
    (each ticker is persisted the moment it is computed, so a restart resumes from
    the store). On a re-run, an already-cached ticker is skipped (no EDGAR call)
    unless ``refresh`` is set. Per-ticker failures are quarantined and the build
    continues; the shared EdgarClient's throttle / 429-Retry-After govern the rate.
    ``limit`` caps the symbols processed (smoke tests); progress is logged AND
    printed to stdout (so it streams through ``tee``) every ``log_every`` tickers."""
    syms = list(symbols)
    if limit is not None:
        syms = syms[:limit]
    log_every = max(1, log_every)
    total = len(syms)
    res = ClassifyResult(n_symbols=total)

    # Announce work up-front, BEFORE the first EDGAR call, so a long build is never
    # silent at startup. Read-only pre-scan of the cache; the loop below is unchanged.
    n_cached_start = 0 if refresh else sum(
        not store.get_data(SIC_FIELD, s, _FAR_FUTURE).empty for s in syms)
    print(f"building universe: {total} symbols to classify "
          f"(cached: {n_cached_start}, to fetch: {total - n_cached_start})", flush=True)

    t0 = time.perf_counter()
    for i, sym in enumerate(syms, 1):
        if not refresh and not store.get_data(SIC_FIELD, sym, _FAR_FUTURE).empty:
            res.n_cached += 1
        else:
            # Fetch under a hard deadline + retry; a timeout/HTTP failure is
            # quarantined (counted as "failed") and the loop continues — it is
            # NEVER written to the store, so a re-run retries it (no cache poison).
            try:
                submissions = _fetch_submissions(
                    sym, client=client, resolver=resolver,
                    timeout=fetch_timeout, retries=fetch_retries, sleep=sleep)
            except (UnknownTickerError, EdgarHTTPError, ClassificationTimeout) as exc:
                res.n_quarantined += 1
                res.quarantine.append({"ticker": sym, "field": SIC_FIELD,
                                       "reason": f"sic_unavailable: {exc}", "vendor": SOURCE})
            else:
                sic = submissions.get("sic") or submissions.get("sicCode")
                first_filing = _earliest_filing_date(submissions)
                if not sic or first_filing is None:
                    # Empty/garbage response -> quarantine, do NOT cache (retryable).
                    res.n_quarantined += 1
                    res.quarantine.append({"ticker": sym, "field": SIC_FIELD,
                                           "reason": "no_sic_or_filing", "vendor": SOURCE})
                else:
                    _cache_one(store, sym, sic, first_filing)  # only genuine verdicts cached
                    res.n_classified += 1
                    if classify_sic(float(int(sic))) is not None:
                        res.n_kept += 1
        if i % log_every == 0 or i == total:
            log.info("classified %d/%d (kept %d hc+tech, skipped %d, failed %d)",
                     i, total, res.n_kept, res.n_cached, res.n_quarantined)
            # Mirror to stdout so the progress streams through the user's `tee`.
            print(f"classified {i}/{total} (kept {res.n_kept}, skipped {res.n_cached}, "
                  f"failed {res.n_quarantined}, cache-hits {res.n_cached})", flush=True)
    print(f"universe build complete: {res.n_kept} names in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
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
