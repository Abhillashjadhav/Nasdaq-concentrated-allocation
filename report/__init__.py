"""Reporting: per-signal lift table + GO/KILL verdict (ARCHITECTURE.md §10).

Emits the per-signal conditional-lift table (CI, p-value), the >=3-year
consistency check, and the final GO / KILL / Partial verdict.

Implemented in PR 13 (ARCHITECTURE.md §9).
"""

from __future__ import annotations


def build_report(*args, **kwargs):
    """Render the per-signal lift table and resolve GO / KILL / Partial."""
    raise NotImplementedError("Report + GO/KILL verdict land in PR 13")
