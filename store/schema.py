"""The universal point-in-time record and its Pandera schema.

Every datum in stockscope is stored as one row of this shape, regardless of
vendor or field. The schema is the IO-level guard (ARCHITECTURE.md §8): every
write through ``store.put_data`` is validated against it before it touches the
database, so a malformed row fails loud at the boundary rather than silently
corrupting the experiment.

Columns
-------
ticker          symbol the datum describes (e.g. "AAPL")
field           what the value measures (e.g. "close")
value           the numeric value
event_date      the period the datum describes (§6)
knowledge_date  when the datum became public — the no-peek key (§6)
source          vendor/provenance (e.g. "yfinance", "stooq")
"""

from __future__ import annotations

import pandera.pandas as pa

COLUMNS = ["ticker", "field", "value", "event_date", "knowledge_date", "source"]

# strict=True: no unexpected columns may sneak in. coerce=True: normalise dtypes
# (dates -> datetime64, value -> float) so callers can pass plain Python values.
RECORD_SCHEMA = pa.DataFrameSchema(
    {
        "ticker": pa.Column(str, nullable=False),
        "field": pa.Column(str, nullable=False),
        "value": pa.Column(float, nullable=False),
        "event_date": pa.Column("datetime64[ns]", nullable=False),
        "knowledge_date": pa.Column("datetime64[ns]", nullable=False),
        "source": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=True,
    ordered=False,
)


def validate(df):
    """Validate a records frame against ``RECORD_SCHEMA`` with ``lazy=True`` so
    every violation is reported at once. Raises ``pandera.errors.SchemaErrors``
    on any failure — callers must not swallow it (CLAUDE.md: no silent drops)."""
    return RECORD_SCHEMA.validate(df, lazy=True)
