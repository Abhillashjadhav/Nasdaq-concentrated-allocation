"""Form 4 insider-transaction adapter (ARCHITECTURE.md §5, §6).

``fetch_insider_buys(ticker)`` lists a company's filings via the SEC submissions
API, fetches each Form 4 ownership XML, and extracts insider transactions —
reusing the ``EdgarClient`` + ``CikResolver`` from ``data.edgar_client``.

Only genuine OPEN-MARKET PURCHASES (transaction code ``P``, non-derivative) feed
the insider cluster signal: they become ``form4_buy_P`` records with
``value`` = the reporting owner's CIK (a stable insider id), ``event_date`` = the
transaction date, ``knowledge_date`` = the filing date (the Form 4 lag, §6). All
other codes — option exercises (``M``), grants (``A``), sells (``S``), etc. — are
routed to ``form4_other_{code}`` fields the signal never reads, so they cannot
inflate the cluster. A ``form4_covered`` marker is written per filing so coverage
is itself point-in-time (a name is "covered" only once its first Form 4 is filed).

A ticker with no Form 4 filings is FLAGGED as a coverage gap, never faked. Field
names match ``signals.insider`` (guarded by a test). Records are written through
``store.put_data`` (Pandera + fail-loud).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field as dc_field

import pandas as pd

import store as store_pkg
from data.edgar_client import CikResolver, EdgarClient, EdgarHTTPError
from store.schema import COLUMNS

log = logging.getLogger(__name__)
SOURCE = "edgar"
BUY_FIELD = "form4_buy_P"        # non-derivative open-market purchases only
COVERAGE_FIELD = "form4_covered"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"


@dataclass
class Form4Txn:
    code: str | None
    date: str | None
    is_derivative: bool
    acquired_disposed: str | None = None  # "A" acquired / "D" disposed


@dataclass
class Form4Result:
    ticker: str
    cik: str
    records: pd.DataFrame
    gaps: list[dict] = dc_field(default_factory=list)
    n_written: int = 0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]  # strip any XML namespace


def _value(elem, name: str) -> str | None:
    """First descendant with local tag ``name``; prefer a nested <value> child."""
    for d in elem.iter():
        if _local(d.tag) == name:
            for c in d:
                if _local(c.tag) == "value":
                    return (c.text or "").strip()
            return (d.text or "").strip()
    return None


def _iter_local(root, name: str):
    for d in root.iter():
        if _local(d.tag) == name:
            yield d


def _looks_like_form4_xml(text) -> bool:
    """Cheap guard against non-XML responses (SEC error/HTML pages) before parsing.
    A Form 4 ownership document always contains the ``ownershipDocument`` element."""
    return isinstance(text, str) and "ownershipDocument" in text


def parse_form4(xml_text: str) -> tuple[str | None, list[Form4Txn]]:
    """Return (reporting owner CIK, transactions) from a Form 4 ownership XML.

    Namespaces are handled explicitly: every tag is compared by its LOCAL name
    (``_local`` strips any ``{namespace}`` prefix), so namespaced and bare Form 4
    documents both parse. Raises ``xml.etree.ElementTree.ParseError`` on malformed
    XML — the caller catches it per filing and quarantines that filing."""
    root = ET.fromstring(xml_text)
    owner_cik = _value(root, "rptOwnerCik")
    txns: list[Form4Txn] = []
    for tag, is_deriv in (("nonDerivativeTransaction", False), ("derivativeTransaction", True)):
        for node in _iter_local(root, tag):
            txns.append(Form4Txn(
                code=_value(node, "transactionCode"),
                date=_value(node, "transactionDate"),
                is_derivative=is_deriv,
                acquired_disposed=_value(node, "transactionAcquiredDisposedCode"),
            ))
    return owner_cik, txns


def _form4_filings(submissions: dict):
    """Yield (accession, filing_date, primary_document) for each Form 4."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    fdates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    for i, form in enumerate(forms):
        if form == "4":
            yield accns[i], fdates[i], docs[i]


def fetch_insider_buys(
    ticker: str,
    *,
    client: EdgarClient | None = None,
    resolver: CikResolver | None = None,
    store=None,
    write: bool = False,
) -> Form4Result:
    """Pull Form 4 open-market purchases for ``ticker`` into universal records."""
    client = client or EdgarClient()
    resolver = resolver or CikResolver(client)
    cik = resolver.resolve(ticker)

    submissions = client.get_json(f"/submissions/CIK{cik}.json")
    filings = list(_form4_filings(submissions))
    gaps: list[dict] = []
    if not filings:
        gaps.append({"ticker": ticker.upper(), "field": COVERAGE_FIELD,
                     "reason": "no_form4_filings", "vendor": SOURCE})
        return Form4Result(ticker.upper(), cik, pd.DataFrame(columns=COLUMNS), gaps, 0)

    rows: list[dict] = []
    cik_int = int(cik)
    for accession, filing_date, primary_doc in filings:
        url = f"{_ARCHIVE}/{cik_int}/{accession.replace('-', '')}/{primary_doc}"
        # Per-filing resilience: a fetch failure, a non-XML response, or malformed
        # XML quarantines THAT filing and is skipped — a ticker's other (good)
        # filings still ingest, and one bad filing never aborts the run.
        def _quarantine(reason):
            gaps.append({"ticker": ticker.upper(), "field": BUY_FIELD,
                         "reason": f"{reason} (accession {accession})", "vendor": SOURCE})
        try:
            text = client.get_text(url)
        except EdgarHTTPError as exc:
            log.warning("form4 fetch failed %s %s: %s", ticker, accession, exc)
            _quarantine(f"form4_fetch_failed: {exc}")
            continue
        if not _looks_like_form4_xml(text):
            log.warning("form4 non-XML response for %s %s", ticker, accession)
            _quarantine("form4_non_xml_response")
            continue
        try:
            owner_cik, txns = parse_form4(text)
        except ET.ParseError as exc:
            log.warning("form4 parse error for %s %s: %s", ticker, accession, exc)
            _quarantine(f"form4_parse_error: {exc}")
            continue
        if owner_cik is None:
            _quarantine("form4_no_owner")
            continue
        kd = pd.Timestamp(filing_date)
        rows.append({"ticker": ticker.upper(), "field": COVERAGE_FIELD, "value": 1.0,
                     "event_date": kd, "knowledge_date": kd, "source": SOURCE})
        for t in txns:
            if t.code is None or t.date is None:
                continue
            # open-market buy (code P, non-derivative, an acquisition) -> the cluster
            # field; every other code/direction is routed elsewhere so it cannot
            # inflate the cluster. (A P-disposal would be malformed; guard anyway.)
            is_buy = t.code == "P" and not t.is_derivative and t.acquired_disposed != "D"
            f = BUY_FIELD if is_buy else f"form4_other_{t.code}"
            rows.append({"ticker": ticker.upper(), "field": f, "value": float(owner_cik),
                         "event_date": pd.Timestamp(t.date), "knowledge_date": kd,
                         "source": SOURCE})

    records = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(
        subset=["field", "event_date", "knowledge_date", "value"]
    ).reset_index(drop=True)
    n_written = 0
    if write and not records.empty:
        n_written = (store or store_pkg).put_data(records)
    return Form4Result(ticker.upper(), cik, records, gaps, n_written)
