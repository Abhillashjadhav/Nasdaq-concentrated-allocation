"""Form 4 insider adapter tests (ARCHITECTURE.md §5, §6).

Golden parse from a hand-built Form 4 XML (no network) proving an A grant and an
M exercise are excluded from form4_buy_P (routed elsewhere), the filing-lag is
respected, the form4_covered marker is emitted, no-Form-4 coverage is flagged,
field names align with signals.insider, EdgarClient.get_text works, and a
network-marked live smoke that skips offline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.edgar_client import CikResolver, EdgarClient
from data.form4 import BUY_FIELD, COVERAGE_FIELD, fetch_insider_buys, parse_form4
from store.store import PITStore

COMPANY_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerId><rptOwnerCik>0001112223</rptOwnerCik></reportingOwnerId></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2021-02-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2021-02-02</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <transactionDate><value>2021-02-03</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""

SUBMISSIONS = {"filings": {"recent": {
    "form": ["4", "8-K"],
    "accessionNumber": ["0001112223-21-000045", "0000000000-21-000001"],
    "filingDate": ["2021-02-03", "2021-01-15"],  # the Form 4 was filed 2021-02-03
    "primaryDocument": ["form4.xml", "doc.htm"],
}}}


class FakeClient:
    def __init__(self, json_responses, text_responses):
        self.json_responses = json_responses
        self.text_responses = text_responses

    def get_json(self, path):
        for k, v in self.json_responses.items():
            if k in path:
                return v
        raise KeyError(path)

    def get_text(self, path):
        for k, v in self.text_responses.items():
            if k in path:
                return v
        raise KeyError(path)


def _fake(submissions=SUBMISSIONS):
    fake = FakeClient(
        {"company_tickers": COMPANY_TICKERS, "submissions": submissions},
        {"form4.xml": FORM4_XML},
    )
    return fake, CikResolver(fake)


def test_parse_form4_extracts_owner_and_codes():
    owner, txns = parse_form4(FORM4_XML)
    assert owner == "0001112223"
    codes = sorted((t.code, t.is_derivative) for t in txns)
    assert codes == [("A", False), ("M", True), ("P", False)]


def test_golden_only_code_P_in_buy_field():
    fake, resolver = _fake()
    res = fetch_insider_buys("AAPL", client=fake, resolver=resolver)

    buys = res.records[res.records["field"] == BUY_FIELD]
    assert len(buys) == 1                                   # only the open-market P
    assert buys.iloc[0]["value"] == 1112223.0              # reporting owner CIK = insider id
    assert pd.Timestamp(buys.iloc[0]["event_date"]) == pd.Timestamp("2021-02-01")
    assert pd.Timestamp(buys.iloc[0]["knowledge_date"]) == pd.Timestamp("2021-02-03")  # filing lag

    fields = set(res.records["field"])
    assert "form4_other_A" in fields and "form4_other_M" in fields  # routed away, can't inflate
    assert (res.records["field"] == COVERAGE_FIELD).sum() == 1


def test_filing_lag_no_leak(tmp_path):
    fake, resolver = _fake()
    store = PITStore(tmp_path / "f4.sqlite")
    fetch_insider_buys("AAPL", client=fake, resolver=resolver, store=store, write=True)

    # transacted 02-01 but filed 02-03 -> invisible until the filing date
    assert store.get_data(BUY_FIELD, "AAPL", date(2021, 2, 2)).empty
    assert len(store.get_data(BUY_FIELD, "AAPL", date(2021, 2, 3))) == 1


def test_no_form4_filings_flagged_not_faked():
    no_f4 = {"filings": {"recent": {"form": ["8-K"], "accessionNumber": ["x"],
                                    "filingDate": ["2021-01-15"], "primaryDocument": ["d.htm"]}}}
    fake, resolver = _fake(no_f4)
    res = fetch_insider_buys("AAPL", client=fake, resolver=resolver)
    assert res.records.empty
    assert any(g["field"] == COVERAGE_FIELD and g["reason"] == "no_form4_filings"
               for g in res.gaps)


def test_fields_align_with_insider_signal():
    from signals.insider import BUY_FIELD as SIG_BUY, COVERAGE_FIELD as SIG_COV
    assert (BUY_FIELD, COVERAGE_FIELD) == (SIG_BUY, SIG_COV)


class _TextResp:
    def __init__(self, text):
        self._t = text

    def raise_for_status(self):
        pass

    @property
    def text(self):
        return self._t


class _TextSession:
    def get(self, url, headers=None, timeout=None):
        return _TextResp("<ownershipDocument/>")


def test_edgar_client_get_text():
    c = EdgarClient(user_agent="t t@e.com", session=_TextSession())
    assert c.get_text("/any") == "<ownershipDocument/>"


@pytest.mark.network
def test_live_smoke_skips_offline():
    try:
        client = EdgarClient(user_agent="stockscope-test test@example.com")
        res = fetch_insider_buys("AAPL", client=client)
    except Exception as exc:  # offline / blocked / 403 -> skip
        pytest.skip(f"EDGAR unavailable: {exc}")
    assert res.cik == "0000320193"
