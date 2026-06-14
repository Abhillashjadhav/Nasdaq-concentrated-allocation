"""Survivorship / coverage report (ARCHITECTURE.md §2, §8).

For a given as-of date, a target ticker list, and the fields we expect, this
reports how many (ticker, field) cells were actually fetched vs. how many were
requested — naming every gap by vendor/ticker/field — and QUARANTINES the
missing cells so downstream code can't silently use a hole as if it were data.

Survivorship angle: a known-delisted ticker must be *detectable*, not silently
absent. If it has data it shows as present; if free sources lack it, it surfaces
as an explicit coverage gap (reason "absent"), never a silent drop. Either way a
dead name is visible, which is the whole point of a survivorship-free harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date


@dataclass
class CoverageReport:
    as_of: date
    fields: list[str]
    tickers: list[str]
    present: list[tuple[str, str]] = dc_field(default_factory=list)
    missing: list[dict] = dc_field(default_factory=list)

    @property
    def n_requested(self) -> int:
        return len(self.fields) * len(self.tickers)

    @property
    def n_fetched(self) -> int:
        return len(self.present)

    @property
    def quarantined(self) -> set[tuple[str, str]]:
        """(ticker, field) cells excluded from downstream use."""
        return {(m["ticker"], m["field"]) for m in self.missing}

    def matrix(self) -> str:
        """Human-readable coverage matrix: rows=tickers, cols=fields, ✓/·."""
        present = {(t, f) for f, t in self.present}
        width = max((len(t) for t in self.tickers), default=6)
        header = " " * (width + 2) + "  ".join(self.fields)
        lines = [
            f"Coverage as of {self.as_of}: fetched {self.n_fetched} of "
            f"{self.n_requested}; {len(self.missing)} missing.",
            header,
        ]
        for t in self.tickers:
            cells = "  ".join(
                ("✓" if (t, f) in present else "·").center(len(f))
                for f in self.fields
            )
            lines.append(f"{t.ljust(width)}  {cells}")
        if self.missing:
            lines.append("Missing:")
            for m in self.missing:
                lines.append(
                    f"  - {m['ticker']}/{m['field']}: {m['reason']} "
                    f"(vendor={m['vendor']})"
                )
        return "\n".join(lines)


def coverage_report(store, fields, tickers, as_of, vendor: str = "store") -> CoverageReport:
    """Probe the store for each (ticker, field) as of ``as_of`` and tally
    present vs. missing. A cell is present iff get_data returns >=1 row."""
    report = CoverageReport(as_of=as_of, fields=list(fields), tickers=list(tickers))
    for f in fields:
        for t in tickers:
            rows = store.get_data(f, t, as_of)
            if rows is not None and not rows.empty:
                report.present.append((f, t))
            else:
                report.missing.append(
                    {"ticker": t, "field": f, "reason": "absent", "vendor": vendor}
                )
    return report
