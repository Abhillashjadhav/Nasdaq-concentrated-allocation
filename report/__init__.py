"""Reporting: per-signal lift table + GO/KILL verdict (ARCHITECTURE.md §3, §10).

Emits the per-signal conditional-lift table (CI, rank-IC), the per-year detail,
the coverage report, and the final GO / MARGINAL / KILL verdict computed exactly
against the §10 thresholds.

Implementation lives in ``report.build_report``.
"""

from __future__ import annotations

from .build_report import (
    Report,
    SignalVerdict,
    build_report,
    evaluate_signal,
    resolve_verdict,
)

__all__ = ["build_report", "Report", "SignalVerdict", "evaluate_signal", "resolve_verdict"]
