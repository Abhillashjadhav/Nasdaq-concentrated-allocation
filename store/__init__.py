"""Point-in-time store and the single data-access chokepoint.

`get_data` is the ONLY way any signal/macro/universe/backtest module is allowed
to read data. It exists to make the no-peek non-negotiable enforceable rather
than trusted: every read filters `knowledge_date <= as_of`.

Implemented in PR 2 (ARCHITECTURE.md §9).
"""

from __future__ import annotations

from datetime import date
from typing import Any


def get_data(field: str, ticker: str, as_of: date) -> Any | None:
    """Return the most recent value of ``field`` for ``ticker`` that was public
    on or before ``as_of`` — i.e. only rows with ``knowledge_date <= as_of``.

    This is the no-peek chokepoint (ARCHITECTURE.md §2, §6). No module may read
    raw data by any other path. ``knowledge_date`` (when a datum became public)
    is distinct from ``event_date`` (the period it describes); filing lags are
    applied at ingest, not here.

    Returns ``None`` when no qualifying row exists.
    """
    raise NotImplementedError("Point-in-time store lands in PR 2")
