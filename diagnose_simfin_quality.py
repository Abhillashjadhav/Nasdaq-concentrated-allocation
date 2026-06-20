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


# --- a faithful SimFin-shaped fixture (free tier: Publish Date arrives empty) ---

def _fixture_frames(ticker="LLY", *, publish_empty=True):
    """Two annual periods with realistic columns/units. ``publish_empty`` mirrors
    SimFin's free bulk, where Publish Date is a (paid) column that ships as NaT."""
    pub1 = pd.NaT if publish_empty else pd.Timestamp("2023-02-20")
    pub2 = pd.NaT if publish_empty else pd.Timestamp("2024-02-20")
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

    # [3] date / knowledge_date filtering (the PIT gate)
    pub_na = {}
    for k, df in frames.items():
        sub = df[df[TICKER_COL].astype(str).str.upper() == t] if present[k] else df.iloc[0:0]
        n = len(sub)
        na = int(sub[PUBLISH_DATE_COL].isna().sum()) if PUBLISH_DATE_COL in sub.columns else n
        pub_na[k] = f"{na}/{n} Publish Date NaT"
    print(f"[3] PIT gate     : {pub_na}")
    records, quarantine, n_skipped = records_from_frames(frames, tickers=[t])
    per_field = {f: int((records.field == f).sum()) for f in FIELDS} if not records.empty else {f: 0 for f in FIELDS}
    print(f"    records_from_frames -> {len(records)} rows, n_skipped={n_skipped}")
    print(f"    per-field record counts: {per_field}")

    # [4] load into the store and run the real quality computation
    store = PITStore(tempfile.mkdtemp() + "/diag.sqlite")
    if not records.empty:
        store.put_data(records)
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
    any_pub_na = any("0/" not in v for v in pub_na.values())
    if records.empty or short:
        if any_pub_na and n_skipped > 0:
            print(f"VERDICT: date/knowledge_date filtering — Publish Date missing, "
                  f"{n_skipped} facts skipped at the PIT gate; short fields={short}")
            return "date/knowledge_date filtering"
        print(f"VERDICT: insufficient periods for fields={short} (not a Publish-Date skip)")
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
