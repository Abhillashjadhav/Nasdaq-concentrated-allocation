"""SQLite-backed point-in-time store.

This module owns the no-peek chokepoint (ARCHITECTURE.md §2, §6). Two operations:

* ``put_data(records)`` — validate against the Pandera record schema, then insert.
  A malformed batch raises loudly; nothing is silently dropped (CLAUDE.md).
* ``get_data(field, ticker, as_of)`` — the single read path. It returns ONLY rows
  with ``knowledge_date <= as_of``, newest ``event_date`` first. There is no code
  path that returns a row past ``as_of`` — the filter is in the SQL itself.

The database file is configurable; it defaults to a gitignored cache dir
(``.data_cache/store.sqlite``) and is deliberately NOT inside the ``data/``
source package.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from .schema import COLUMNS, validate

DEFAULT_DB_PATH = Path(".data_cache/store.sqlite")
_TABLE = "pit_records"


def _as_of_ts(as_of) -> pd.Timestamp:
    """Normalise a date/datetime/str into a day-resolution Timestamp."""
    return pd.Timestamp(as_of).normalize()


class PITStore:
    """A point-in-time store backed by a single SQLite table."""

    def __init__(self, db_path: str | os.PathLike | None = None):
        self.db_path = Path(
            db_path or os.environ.get("STOCKSCOPE_DB", DEFAULT_DB_PATH)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    ticker         TEXT NOT NULL,
                    field          TEXT NOT NULL,
                    value          REAL NOT NULL,
                    event_date     TEXT NOT NULL,
                    knowledge_date TEXT NOT NULL,
                    source         TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_lookup "
                f"ON {_TABLE} (field, ticker, knowledge_date)"
            )

    def put_data(self, records: pd.DataFrame) -> int:
        """Validate ``records`` against the schema and insert them.

        Raises ``pandera.errors.SchemaErrors`` on any validation failure (no
        partial / silent inserts). Returns the number of rows written.
        """
        validated = validate(records)
        rows = validated.copy()
        # store dates as ISO day strings so lexical compare == chronological compare
        rows["event_date"] = pd.to_datetime(rows["event_date"]).dt.strftime("%Y-%m-%d")
        rows["knowledge_date"] = (
            pd.to_datetime(rows["knowledge_date"]).dt.strftime("%Y-%m-%d")
        )
        rows = rows[COLUMNS]
        with self._connect() as conn:
            rows.to_sql(_TABLE, conn, if_exists="append", index=False)
        return len(rows)

    def get_data(self, field: str, ticker: str, as_of) -> pd.DataFrame:
        """Return all point-in-time records for ``(field, ticker)`` that were
        public on or before ``as_of`` — i.e. ``knowledge_date <= as_of`` — newest
        ``event_date`` first. Returns an empty frame when none qualify.

        This is the no-peek chokepoint: the ``knowledge_date <= as_of`` predicate
        is applied in SQL and cannot be bypassed by callers.
        """
        cutoff = _as_of_ts(as_of).strftime("%Y-%m-%d")
        with self._connect() as conn:
            df = pd.read_sql_query(
                f"""
                SELECT {", ".join(COLUMNS)}
                FROM {_TABLE}
                WHERE field = ? AND ticker = ? AND knowledge_date <= ?
                ORDER BY event_date DESC
                """,
                conn,
                params=(field, ticker, cutoff),
                parse_dates=["event_date", "knowledge_date"],
            )
        # Pandera guard on the READ boundary too (ARCHITECTURE.md §8): anything
        # leaving the store satisfies the universal record schema. Empty results
        # are a valid (no qualifying rows) answer, not a schema violation.
        if not df.empty:
            df = validate(df)
        return df


# --- module-level default store, preserving the §6 chokepoint signature -------

_default_store: PITStore | None = None


def _store() -> PITStore:
    global _default_store
    if _default_store is None:
        _default_store = PITStore()
    return _default_store


def get_data(field: str, ticker: str, as_of: date) -> pd.DataFrame:
    """Chokepoint read against the default store (ARCHITECTURE.md §6)."""
    return _store().get_data(field, ticker, as_of)


def put_data(records: pd.DataFrame) -> int:
    """Validated write to the default store."""
    return _store().put_data(records)
