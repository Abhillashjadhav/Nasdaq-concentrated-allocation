"""SimFin bulk fundamentals adapter (ARCHITECTURE.md §5, §6).

Replaces per-ticker EDGAR fetching with SimFin's bulk US-fundamentals datasets.
EDGAR's companyfacts API is rate-limited per IP, so a 1,800-name run throttles and
the quality signal comes back n/a for almost everything. SimFin ships the same
figures as a handful of bulk downloads (~thousands of companies, ~20 years), so
ONE pull populates the whole universe.

Point-in-time correctness (the critical part)
---------------------------------------------
``event_date``     = SimFin "Report Date"  (fiscal period end)
``knowledge_date`` = SimFin "Publish Date" (when the figure first became public)

Using Publish Date — not the period end — as the no-peek key means the store's
``knowledge_date <= as_of`` filter keeps a period that has ended but not yet been
published correctly invisible, so filing lag is respected (a test proves it).

SimFin's FREE bulk ships Publish Date empty. When it is missing we do NOT drop the
fact (that left quality n/a for the whole universe); instead ``knowledge_date`` =
period end + a conservative filing lag (FY ~90d, quarter ~45d). That key is always
strictly AFTER the period end, so no future row can leak — no-peek still holds.

Network boundary
----------------
The download happens HERE, at ingest, and is cached under ``.data_cache/simfin/``
(gitignored) so re-runs are a no-op unless ``refresh=True``. Parsing + loading are
pure and offline-testable: ``records_from_frames`` takes plain DataFrames, and
``load_simfin_fundamentals`` accepts an injected ``frames_loader`` so no network or
``simfin`` package is needed in tests. No network ever touches signal compute.

The nine target field names match ``signals.quality.FIELDS`` so this adapter feeds
the quality signal with no change to its math (a drift test guards the alignment).
The API key is configuration read from ``STOCKSCOPE_SIMFIN_API_KEY`` (env only,
never committed).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import pandas as pd

from store.schema import COLUMNS

log = logging.getLogger(__name__)

SOURCE = "simfin"
API_KEY_ENV = "STOCKSCOPE_SIMFIN_API_KEY"
DEFAULT_CACHE_DIR = Path(".data_cache/simfin")

# SimFin's per-row identity + point-in-time columns (common to every dataset).
TICKER_COL = "Ticker"
REPORT_DATE_COL = "Report Date"
PUBLISH_DATE_COL = "Publish Date"
FISCAL_PERIOD_COL = "Fiscal Period"

# SimFin's FREE bulk datasets ship "Publish Date" (a paid-tier column) empty, while
# "Report Date" (the fiscal period end) is always present. Rather than drop every
# such fact — which leaves the quality signal n/a for the whole universe — we fall
# back to a CONSERVATIVE knowledge_date = period end + a filing lag. This pushes the
# no-peek key strictly AFTER the period closed (never before), so no future row can
# leak (ARCHITECTURE.md §6: filing lags applied at ingest). Lags mirror SEC norms.
_ANNUAL_LAG = pd.Timedelta(days=90)     # 10-K: filed up to ~60-90d after FY end
_QUARTERLY_LAG = pd.Timedelta(days=45)  # 10-Q: ~40-45d after quarter end
_DEFAULT_LAG = pd.Timedelta(days=75)    # period type unknown -> conservative middle


def _filing_lag(fiscal_period) -> pd.Timedelta:
    """Knowledge-date lag to apply when SimFin omits Publish Date, keyed off the
    SimFin Fiscal Period ("FY" vs "Q1".."Q4"). Always >= the period end."""
    if fiscal_period is None or pd.isna(fiscal_period):
        return _DEFAULT_LAG
    p = str(fiscal_period).strip().upper()
    if p in ("FY", "ANNUAL", "Y"):
        return _ANNUAL_LAG
    if p.startswith("Q") or p in ("H1", "H2", "9M"):
        return _QUARTERLY_LAG
    return _DEFAULT_LAG


# target field -> (dataset, candidate SimFin columns in fallback priority order).
# Keys MUST equal signals.quality.FIELDS (guarded by a test).
FIELD_MAP: dict[str, tuple[str, list[str]]] = {
    "revenue": ("income", ["Revenue"]),
    "gross_profit": ("income", ["Gross Profit"]),
    "net_income": ("income", ["Net Income", "Net Income (Common)"]),
    "shares_outstanding": ("income", ["Shares (Diluted)", "Shares (Basic)"]),
    "cfo": ("cashflow", ["Net Cash from Operating Activities"]),
    "total_assets": ("balance", ["Total Assets"]),
    "current_assets": ("balance", ["Total Current Assets"]),
    "current_liabilities": ("balance", ["Total Current Liabilities"]),
    "long_term_debt": ("balance", ["Long Term Debt"]),
}


class SimFinConfigError(RuntimeError):
    """Raised for setup problems (missing package or API key) — not a data gap."""


@dataclass
class SimFinResult:
    n_written: int = 0
    n_records: int = 0
    n_skipped: int = 0  # facts dropped for a missing value/date (no PIT key)
    quarantine: list[dict] = dc_field(default_factory=list)  # evals.coverage shape


def records_from_frames(
    frames: dict[str, pd.DataFrame], *, tickers=None
) -> tuple[pd.DataFrame, list[dict], int]:
    """Map SimFin income/balance/cashflow frames to universal PIT records.

    Pure and deterministic. Returns ``(records, quarantine, n_skipped)``:
      * ``records``    — a validated-shape frame (store.schema.COLUMNS)
      * ``quarantine`` — tickers requested via ``tickers`` but absent from SimFin
      * ``n_skipped``  — facts dropped because value/Report Date/Publish Date was
        missing (counted, never silently written as zero)
    """
    present: set[str] = set()
    for df in frames.values():
        if df is not None and not df.empty and TICKER_COL in df.columns:
            present.update(df[TICKER_COL].astype(str).str.upper())
    wanted = None if tickers is None else {t.upper() for t in tickers}

    rows: list[dict] = []
    n_skipped = 0
    n_lagged = 0
    for target_field, (dataset_key, candidates) in FIELD_MAP.items():
        df = frames.get(dataset_key)
        if df is None or df.empty:
            continue
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue  # this dataset lacks the column; other fields still load
        for _, r in df.iterrows():
            tkr = str(r[TICKER_COL]).upper()
            if wanted is not None and tkr not in wanted:
                continue
            val, report = r.get(col), r.get(REPORT_DATE_COL)
            if pd.isna(val) or pd.isna(report):
                n_skipped += 1  # no value or no period end -> genuinely unusable
                continue
            publish = r.get(PUBLISH_DATE_COL)
            if pd.isna(publish):
                # Publish Date missing (free tier): derive a no-peek-safe knowledge
                # date = period end + filing lag rather than discarding the fact.
                knowledge = pd.Timestamp(report) + _filing_lag(r.get(FISCAL_PERIOD_COL))
                n_lagged += 1
            else:
                knowledge = pd.Timestamp(publish)
            rows.append({
                "ticker": tkr, "field": target_field, "value": float(val),
                "event_date": pd.Timestamp(report),
                "knowledge_date": knowledge, "source": SOURCE,
            })
    if n_lagged:
        log.info("SimFin: %d facts had no Publish Date; used Report Date + filing "
                 "lag as a conservative knowledge_date (no-peek preserved)", n_lagged)

    records = pd.DataFrame(rows, columns=COLUMNS)
    if not records.empty:
        # Restatements kept: distinct (period, publish, value) survive; only EXACT
        # duplicate rows are collapsed — mirrors the EDGAR adapter.
        records = records.drop_duplicates(
            subset=["ticker", "field", "event_date", "knowledge_date", "value"]
        ).reset_index(drop=True)

    quarantine: list[dict] = []
    if wanted is not None:
        for tkr in sorted(wanted - present):
            quarantine.append({"ticker": tkr, "field": "fundamentals",
                               "reason": "absent_from_simfin", "vendor": SOURCE})
    return records, quarantine, n_skipped


def _download_frames(*, api_key, cache_dir, refresh, variants) -> dict[str, pd.DataFrame]:
    """Download (or read cached) SimFin bulk datasets via the ``simfin`` package.

    Network happens HERE only. ``simfin`` caches under ``cache_dir`` itself, so a
    second run re-reads the local CSVs; ``refresh=True`` forces a fresh pull.
    """
    try:
        import simfin as sf
    except ImportError as exc:  # setup problem, fail loud (not a per-ticker gap)
        raise SimFinConfigError(
            "the 'simfin' package is required for SimFin ingest (pip install simfin)"
        ) from exc

    key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise SimFinConfigError(
            f"SimFin API key required; set {API_KEY_ENV} (free key at simfin.com)"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(key)
    sf.set_data_dir(str(cache_dir))
    refresh_days = 0 if refresh else 30  # 0 -> always re-download

    loaders = {"income": sf.load_income, "balance": sf.load_balance,
               "cashflow": sf.load_cashflow}
    frames: dict[str, pd.DataFrame] = {}
    for name, fn in loaders.items():
        parts = [fn(variant=v, market="us", refresh_days=refresh_days).reset_index()
                 for v in variants]
        frames[name] = pd.concat(parts, ignore_index=True)
    return frames


def load_simfin_fundamentals(
    store,
    *,
    api_key=None,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh: bool = False,
    variants=("annual", "quarterly"),
    tickers=None,
    frames_loader=None,
) -> SimFinResult:
    """Download SimFin fundamentals (or read the cache) and load them into the PIT
    store via the validated ``put_data`` chokepoint.

    ``frames_loader`` is injected in tests to bypass network/``simfin``. Tickers in
    ``tickers`` that SimFin doesn't cover are quarantined (never crash); the count
    is logged.
    """
    loader = frames_loader or _download_frames
    frames = loader(api_key=api_key, cache_dir=cache_dir, refresh=refresh, variants=variants)

    records, quarantine, n_skipped = records_from_frames(frames, tickers=tickers)
    n_written = store.put_data(records) if not records.empty else 0
    log.info(
        "SimFin fundamentals: %d records written, %d tickers quarantined, "
        "%d facts skipped (missing value/date)",
        n_written, len(quarantine), n_skipped,
    )
    return SimFinResult(n_written=n_written, n_records=len(records),
                        n_skipped=n_skipped, quarantine=quarantine)
