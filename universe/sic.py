"""SIC-code sector classification for the Nasdaq healthcare + technology universe.

The investable universe (ARCHITECTURE.md §1, §3) is US-listed technology +
healthcare names. We classify a company by its SEC SIC code (Standard Industrial
Classification) into "technology" / "healthcare" / None (excluded). The ranges are
defined explicitly here — broad but documented — so the universe is reproducible
and auditable, not a hand-picked list.
"""

from __future__ import annotations

# (low, high) inclusive SIC ranges. Sources: SEC SIC code list.
TECH_SIC_RANGES = [
    (3570, 3579),  # computer & office equipment
    (3661, 3679),  # communications equipment, electronic components, semiconductors
    (7370, 7379),  # computer programming, data processing, prepackaged software, IT services
]
HEALTHCARE_SIC_RANGES = [
    (2833, 2836),  # medicinal chemicals, pharmaceutical preparations, biological products
    (3826, 3826),  # laboratory analytical instruments
    (3829, 3829),  # measuring & controlling devices
    (3841, 3851),  # surgical/medical/dental/ophthalmic instruments & supplies
    (8000, 8099),  # health services
]

TECHNOLOGY = "technology"
HEALTHCARE = "healthcare"


def _in_any(sic: int, ranges) -> bool:
    return any(lo <= sic <= hi for lo, hi in ranges)


def classify_sic(sic) -> str | None:
    """Return ``"technology"`` / ``"healthcare"`` for an in-scope SIC code, else
    ``None`` (the name is excluded from the universe). Non-numeric/absent -> None."""
    if sic is None or sic == "":
        return None
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    if _in_any(code, TECH_SIC_RANGES):
        return TECHNOLOGY
    if _in_any(code, HEALTHCARE_SIC_RANGES):
        return HEALTHCARE
    return None
