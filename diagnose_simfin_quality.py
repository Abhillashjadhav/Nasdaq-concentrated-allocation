"""Diagnose where the SimFin -> quality chain returns null.

For one large-cap (LLY / AAPL), this walks the EXACT pipeline the run uses —
``records_from_frames`` -> ``store.put_data`` -> ``store.get_data`` ->
``quality_score`` — and prints which of four failure modes is responsible:

    1. ticker-key mismatch   (the name is absent from the SimFin frames)
    2. column-name mismatch  (a FIELD_MAP column is not in the dataset)
    3. date/knowledge_date   (Publish Date missing -> facts skipped at the PIT gate)
    4. units / sign          (a non-positive denominator -> invalid_fundamentals)

Run on your machine against the real cache (needs STOCKSCOPE_SIMFIN_API_KEY set
and .data_cache/simfin present):

    python diagnose_simfin_quality.py            # uses the real cache if available
    python diagnose_simfin_quality.py --fixture  # force the offline reproduction

With no cache/key it falls back to a faithful SimFin-shaped fixture that
reproduces the free-tier "Publish Date is empty" situation.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date

import pandas as pd

from data.simfin_client import (
    FIELD_MAP,
    PUBLISH_DATE_COL,
    REPORT_DATE_COL,
    TICKER_COL,
    records_from_frames,
)
from signals.quality import ANCHOR, FIELDS, MIN_PERIODS, quality_score
from store.store import PITStore

AS_OF = date(2024, 6, 30)


def _coerce(value):
    """Self-contained date coercion (does not import the adapter, so this diagnostic
    runs unchanged on old and new code). Returns a Timestamp or None — None for NaN,
    NaT, None, and the EMPTY STRINGS that SimFin's free tier ships (which .isna()
    does not flag)."""
    if isinstance(value, str):
        if value.strip() == "":
            return None
    elif value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else pd.Timestamp(ts)


# --- a faithful SimFin-shaped fixture (free tier: Publish Date ships as "" ) -----

def _fixture_frames(ticker="LLY", *, publish_empty=True):
    """Two annual periods with realistic columns/units. ``publish_empty`` mirrors
    SimFin's free bulk, where Publish Date ships as an EMPTY STRING (not NaT) — the
    exact shape that slipped past the PR #33 fallback on the real ingest path."""
    pub1 = "" if publish_empty else pd.Timestamp("2023-02-20")
    pub2 = "" if publish_empty else pd.Timestamp("2024-02-20")
    inc = pd.DataFrame([
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2022-12-31"),
         "Publish Date": pub1, "Revenue": 28_500e6, "Gross Profit": 21_000e6,
         "Net Income": 6_200e6, "Shares (Diluted)": 955e6},
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2023-12-31"),
         "Publish Date": pub2, "Revenue": 34_100e6, "Gross Profit": 26_900e6,
         "Net Income": 5_240e6, "Shares (Diluted)": 950e6},
    ])
    bal = pd.DataFrame([
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2022-12-31"),
         "Publish Date": pub1, "Total Assets": 49_000e6, "Total Current Assets": 23_000e6,
         "Total Current Liabilities": 19_000e6, "Long Term Debt": 14_000e6},
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2023-12-31"),
         "Publish Date": pub2, "Total Assets": 64_000e6, "Total Current Assets": 27_000e6,
         "Total Current Liabilities": 26_000e6, "Long Term Debt": 18_000e6},
    ])
    cf = pd.DataFrame([
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2022-12-31"),
         "Publish Date": pub1, "Net Cash from Operating Activities": 7_600e6},
        {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": pd.Timestamp("2023-12-31"),
         "Publish Date": pub2, "Net Cash from Operating Activities": 4_240e6},
    ])
    return {"income": inc, "balance": bal, "cashflow": cf}


def _real_frames():
    """Read the cached SimFin datasets exactly as the run does. Returns None if the
    cache/key/package isn't available (so the caller falls back to the fixture)."""
    try:
        from data.simfin_client import _download_frames
        return _download_frames(api_key=None, cache_dir=".data_cache/simfin",
                                refresh=False, variants=("annual", "quarterly"))
    except Exception as exc:  # noqa: BLE001 — diagnostic, report and fall back
        print(f"  (real cache unavailable: {type(exc).__name__}: {exc})")
        return None


def diagnose(frames, ticker, as_of=AS_OF):
    """Print the four-mode diagnosis for ``ticker`` and return a verdict string."""
    t = ticker.upper()
    print(f"\n===== DIAGNOSE {t} (as_of {as_of}) =====")

    # [1] ticker-key mismatch
    present = {k: (df is not None and not df.empty and TICKER_COL in df.columns
                   and t in set(df[TICKER_COL].astype(str).str.upper()))
               for k, df in frames.items()}
    print(f"[1] ticker-key   : present in { {k: v for k, v in present.items()} }")
    if not any(present.values()):
        print("VERDICT: ticker-key mismatch — name absent from every SimFin frame")
        return "ticker-key mismatch"

    # [2] column-name mismatch
    missing_cols = []
    for fld, (dsk, cands) in FIELD_MAP.items():
        df = frames.get(dsk)
        if df is None or not any(c in df.columns for c in cands):
            missing_cols.append(f"{fld}<-{cands} in '{dsk}'")
    print(f"[2] column-name  : missing = {missing_cols or 'none'}")
    if missing_cols:
        print(f"VERDICT: column-name mismatch — {missing_cols}")
        return "column-name mismatch"

    # [3] date / knowledge_date filtering (the PIT gate). Count PERIODS that yield a
    # usable PIT record — a row is usable only if records_from_frames emits a record
    # with a valid (non-NaT) knowledge_date for its period. We also show blank Publish
    # Dates via a robust check (catches the empty STRINGS that .isna() misses).
    records, quarantine, n_skipped = records_from_frames(frames, tickers=[t])
    good_event_dates = (set(records.loc[records.knowledge_date.notna(), "event_date"])
                        if not records.empty else set())
    usable, blank_pub = {}, {}
    for k, df in frames.items():
        sub = df[df[TICKER_COL].astype(str).str.upper() == t] if present[k] else df.iloc[0:0]
        n = len(sub)
        reps = [_coerce(d) for d in sub[REPORT_DATE_COL]] if REPORT_DATE_COL in sub.columns else []
        u = sum(1 for d in reps if d is not None and pd.Timestamp(d) in good_event_dates)
        nblank = sum(1 for v in (sub[PUBLISH_DATE_COL] if PUBLISH_DATE_COL in sub.columns else [])
                     if _coerce(v) is None)
        usable[k] = f"{u}/{n} usable"
        blank_pub[k] = f"{nblank}/{n} blank publish"
    print(f"[3] PIT gate     : usable={usable}")
    print(f"    publish blanks (robust, incl. empty strings): {blank_pub}")
    per_field = {f: int((records.field == f).sum()) for f in FIELDS} if not records.empty else {f: 0 for f in FIELDS}
    print(f"    records_from_frames -> {len(records)} rows, n_skipped={n_skipped}")
    print(f"    per-field record counts: {per_field}")

    # [4] load into the store and run the real quality computation. Guarded: pre-fix,
    # blank-publish rows carry NaT knowledge_date and the schema rejects them — that
    # rejection is itself the symptom, so report it instead of crashing.
    store = PITStore(tempfile.mkdtemp() + "/diag.sqlite")
    if not records.empty:
        try:
            store.put_data(records)
        except Exception as exc:  # noqa: BLE001 — diagnostic surfaces, never aborts
            print(f"    put_data REJECTED records: {type(exc).__name__} "
                  f"(NaT knowledge_date from blank Publish Date)")
    store_counts = {f: len(store.get_data(f, t, as_of)) for f in FIELDS}
    print(f"[4] store rows   : {store_counts}  (each needs >= {MIN_PERIODS})")
    sample = {f: (store.get_data(f, t, as_of)["value"].iloc[0] if store_counts[f] else None)
              for f in (ANCHOR, "revenue", "net_income")}
    print(f"    unit sanity  : {sample}")

    q = quality_score(t, as_of, store=store)
    verdict_q = q.reason if q.insufficient_data else f"OK score={q.score:.1f} F={q.f_score}"
    print(f"[5] quality_score: {verdict_q}")

    # --- map the observed failure onto one of the four modes ---
    if not q.insufficient_data:
        print("VERDICT: OK — quality populates")
        return "ok"
    short = [f for f, n in store_counts.items() if n < MIN_PERIODS]
    any_blank = any(not v.startswith("0/") for v in blank_pub.values())
    nat_records = (not records.empty) and bool(records["knowledge_date"].isna().any())
    if records.empty or short:
        if any_blank and (nat_records or n_skipped > 0 or not records.empty):
            print("VERDICT: date/knowledge_date filtering — Publish Date blank (empty "
                  "string / NaT) -> knowledge_date unusable; facts dropped at the PIT "
                  f"gate; short fields={short}")
            return "date/knowledge_date filtering"
        print(f"VERDICT: insufficient periods for fields={short} (not a Publish-Date issue)")
        return "insufficient periods"
    if q.reason == "invalid_fundamentals":
        print("VERDICT: units/sign — a non-positive denominator")
        return "units/sign"
    print(f"VERDICT: unclassified — quality reason={q.reason}")
    return q.reason or "unknown"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    use_fixture = "--fixture" in argv
    frames = None if use_fixture else _real_frames()
    if frames is None:
        print("Using offline FIXTURE (faithful SimFin shape; Publish Date empty like free tier).")
        for tkr in ("LLY", "AAPL"):
            diagnose(_fixture_frames(tkr), tkr)
    else:
        print("Using REAL .data_cache/simfin datasets.")
        for tkr in ("LLY", "AAPL"):
            diagnose(frames, tkr)


if __name__ == "__main__":
    main()
