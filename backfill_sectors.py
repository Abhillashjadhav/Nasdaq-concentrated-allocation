"""Backfill sector classification for tickers that have fundamentals but no sector.

Problem this solves
-------------------
The ranking funnel (``report.ranking``) only scores names that are IN the universe,
and universe membership is a cached ``sector`` record in the point-in-time store
(``universe.hc_tech``). If an earlier ingest wrote fundamentals for ~1,600 tickers
but ``classify_and_cache`` only ran over a small subset (e.g. a ``--universe-limit``
smoke run), most names have revenue/quality rows yet no ``sector`` row — so the
ranker silently excludes them.

This script finds every ticker that has at least one fundamental field in
``pit_records`` but no ``sector`` row, then runs ``classify_and_cache`` over exactly
that set. Classification reads SimFin's company+industry reference (cached zips under
``.data_cache/simfin/`` — no network unless ``--refresh``), maps each name's sector
to technology/healthcare, and caches the in-scope ones point-in-time. Out-of-scope
names (financials, energy, …) are correctly left unclassified; names absent from the
SimFin reference are quarantined, never crash.

Usage
-----
    uv run python backfill_sectors.py --db data/store.db

Then re-run the ranking; the previously-excluded names now have a sector row and are
scored.
"""

from __future__ import annotations

import argparse
import sqlite3

from signals.quality import FIELDS as QUALITY_FIELDS
from store.store import PITStore
from universe.hc_tech import SECTOR_FIELD, classify_and_cache


def tickers_with_fundamentals_missing_sector(db_path: str) -> list[str]:
    """Distinct tickers that have at least one quality/fundamental field in the store
    but NO ``sector`` row yet. These are the names the ranker is silently excluding."""
    placeholders = ",".join("?" for _ in QUALITY_FIELDS)
    sql = f"""
        SELECT DISTINCT ticker FROM pit_records
        WHERE field IN ({placeholders})
          AND ticker NOT IN (
              SELECT ticker FROM pit_records WHERE field = ?
          )
        ORDER BY ticker
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(sql, (*QUALITY_FIELDS, SECTOR_FIELD)).fetchall()
    finally:
        con.close()
    return [t for (t,) in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the point-in-time store")
    parser.add_argument("--refresh", action="store_true",
                        help="force re-download of the SimFin reference (default: use cache)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of tickers classified (smoke test)")
    args = parser.parse_args(argv)

    store = PITStore(args.db)
    candidates = tickers_with_fundamentals_missing_sector(args.db)
    print(f"tickers with fundamentals but no sector row: {len(candidates)}")
    if not candidates:
        print("nothing to backfill — every fundamentals ticker already has a sector row")
        return 0

    res = classify_and_cache(candidates, store=store, refresh=args.refresh,
                             limit=args.limit)
    print(
        f"classified={res.n_classified} kept(hc+tech)={res.n_kept} "
        f"quarantined={res.n_quarantined} already-cached={res.n_cached}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
