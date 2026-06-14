"""EDGAR fundamentals adapter (ARCHITECTURE.md §5, §6).

``fetch_fundamentals(ticker)`` pulls the company's XBRL ``companyfacts`` from
EDGAR and extracts the nine us-gaap concepts the quality signal needs, mapping
each fact to a universal point-in-time record and writing it through the existing
``store.put_data`` (so Pandera validation + fail-loud apply).

Point-in-time: ``event_date`` = the period end (``fact["end"]``), ``knowledge_date``
= the filing date (``fact["filed"]``). This is the large fundamentals lag — a Q4
ending Dec-31 is invisible until its 10-K is filed weeks/months later.

Restatements (critical): companyfacts holds MULTIPLE values for the same period
from different filings (original 10-Q, later 10-K, amendments), each with its own
``filed`` date. We keep every ``(value, filed)`` as a SEPARATE record and never
collapse to the latest — the store's ``knowledge_date <= as_of`` filter then
returns the version that was known as-of any query date (the original before the
restatement was filed).

Coverage: each us-gaap concept has several possible tags; we try a documented
fallback list. When a company exposes none of them, that concept is FLAGGED as a
coverage gap (same shape evals.coverage uses) — never silently written as zero.

The nine target field names match ``signals.quality.FIELDS`` so this adapter feeds
the quality signal directly (a test guards against drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import pandas as pd

import store as store_pkg
from data.edgar_client import CikResolver, EdgarClient
from store.schema import COLUMNS

SOURCE = "edgar"

# (target field, us-gaap tags in fallback priority order, XBRL unit). us-gaap
# exposes several tags per economic concept; we take the first present.
CONCEPTS = [
    ("net_income", ["NetIncomeLoss", "ProfitLoss"], "USD"),
    ("cfo", ["NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "USD"),
    ("total_assets", ["Assets"], "USD"),
    ("long_term_debt", ["LongTermDebtNoncurrent", "LongTermDebt"], "USD"),
    ("current_assets", ["AssetsCurrent"], "USD"),
    ("current_liabilities", ["LiabilitiesCurrent"], "USD"),
    ("shares_outstanding", ["CommonStockSharesOutstanding",
                            "EntityCommonStockSharesOutstanding"], "shares"),
    ("gross_profit", ["GrossProfit"], "USD"),
    ("revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "SalesRevenueNet"], "USD"),
]


@dataclass
class FundamentalsResult:
    ticker: str
    cik: str
    records: pd.DataFrame
    gaps: list[dict] = dc_field(default_factory=list)
    n_written: int = 0


def _entries_for(usgaap: dict, tags: list[str], unit: str):
    """Return the unit entries for the first present tag, or None."""
    for tag in tags:
        node = usgaap.get(tag)
        if node and unit in node.get("units", {}):
            return node["units"][unit]
    return None


def fetch_fundamentals(
    ticker: str,
    *,
    client: EdgarClient | None = None,
    resolver: CikResolver | None = None,
    store=None,
    write: bool = False,
) -> FundamentalsResult:
    """Pull and parse EDGAR fundamentals for ``ticker``. Optionally write to the
    store. Reads via the EDGAR client; writes via the store's validated put_data."""
    client = client or EdgarClient()
    resolver = resolver or CikResolver(client)
    cik = resolver.resolve(ticker)  # fails loud on unknown ticker

    facts = client.get_json(f"/api/xbrl/companyfacts/CIK{cik}.json")
    usgaap = facts.get("facts", {}).get("us-gaap", {})

    rows: list[dict] = []
    gaps: list[dict] = []
    for field, tags, unit in CONCEPTS:
        entries = _entries_for(usgaap, tags, unit)
        if entries is None:
            gaps.append({"ticker": ticker.upper(), "field": field,
                         "reason": "no_edgar_concept", "vendor": SOURCE})
            continue
        for e in entries:
            if e.get("val") is None or e.get("end") is None or e.get("filed") is None:
                continue
            rows.append({
                "ticker": ticker.upper(), "field": field, "value": float(e["val"]),
                "event_date": pd.Timestamp(e["end"]),
                "knowledge_date": pd.Timestamp(e["filed"]), "source": SOURCE,
            })

    records = pd.DataFrame(rows, columns=COLUMNS)
    if not records.empty:
        # Restatements kept: distinct (value, filed) per period survive; only
        # EXACT duplicate rows (same period, filing, and value) are collapsed.
        records = records.drop_duplicates(
            subset=["field", "event_date", "knowledge_date", "value"]
        ).reset_index(drop=True)

    n_written = 0
    if write and not records.empty:
        n_written = (store or store_pkg).put_data(records)

    return FundamentalsResult(ticker.upper(), cik, records, gaps, n_written)
