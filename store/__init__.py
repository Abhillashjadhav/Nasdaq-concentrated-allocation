"""Point-in-time store and the single data-access chokepoint.

``get_data`` is the ONLY way any signal/macro/universe/backtest module is allowed
to read data. It exists to make the no-peek non-negotiable enforceable rather
than trusted: every read filters ``knowledge_date <= as_of`` (ARCHITECTURE.md
§2, §6). ``put_data`` is the only write path; it validates every batch against
the Pandera record schema before insert.

The implementation lives in ``store.store`` (SQLite-backed); this package
re-exports the chokepoint so callers depend on ``store``, not on the backend.
"""

from __future__ import annotations

from .schema import RECORD_SCHEMA, validate
from .store import PITStore, get_data, put_data

__all__ = ["get_data", "put_data", "PITStore", "RECORD_SCHEMA", "validate"]
