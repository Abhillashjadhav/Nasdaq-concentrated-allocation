"""SimFin bulk fundamentals adapter tests (ARCHITECTURE.md §5, §6).

Evals-first (CLAUDE.md): these guard the adapter that replaces per-ticker EDGAR
fetching with SimFin's bulk download. All offline — hand-built mini SimFin frames,
no network and no `simfin` package needed (the loader is injected).

Covered:
  * field-map alignment — the adapter feeds every field quality reads (drift guard)
  * PIT mapping — SimFin "Publish Date" becomes knowledge_date, "Report Date" the
    event_date, so filing lag is respected
  * NO-PEEK (eval a) — a figure whose Publish Date is after the as_of is invisible
  * COVERAGE (eval b) — quality populates (non-n/a) for a large-cap fixture once
    SimFin data is loaded
  * missing-ticker quarantine (never crash) and fail-loud skip of unusable facts
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.simfin_client import (
    FIELD_MAP,
    SOURCE,
    load_simfin_fundamentals,
    records_from_frames,
)
from signals.quality import FIELDS, quality_score
from store.store import PITStore

# (period end, publish date) — the latest quarter is filed weeks after it ends,
# which is exactly what the no-peek eval below exploits.
PRIOR = ("2020-09-30", "2020-10-30")
LATEST = ("2020-12-31", "2021-02-15")


def _income(t, report, publish, revenue, gross_profit, net_income, shares):
    return {"Ticker": t, "Report Date": report, "Publish Date": publish,
            "Revenue": revenue, "Gross Profit": gross_profit,
            "Net Income": net_income, "Shares (Diluted)": shares}


def _balance(t, report, publish, ta, ca, cl, ltd):
    return {"Ticker": t, "Report Date": report, "Publish Date": publish,
            "Total Assets": ta, "Total Current Assets": ca,
            "Total Current Liabilities": cl, "Long Term Debt": ltd}


def _cashflow(t, report, publish, cfo):
    return {"Ticker": t, "Report Date": report, "Publish Date": publish,
            "Net Cash from Operating Activities": cfo}


def _frames(tickers=("AAPL",)):
    """Two fully-scoreable quarters per ticker (positive denominators in both)."""
    inc, bal, cf = [], [], []
    for t in tickers:
        inc.append(_income(t, *PRIOR, 900, 400, 100, 170))
        inc.append(_income(t, *LATEST, 1000, 480, 150, 160))
        bal.append(_balance(t, *PRIOR, 1000, 450, 200, 350))
        bal.append(_balance(t, *LATEST, 1100, 500, 200, 300))
        cf.append(_cashflow(t, *PRIOR, 180))
        cf.append(_cashflow(t, *LATEST, 200))
    return {"income": pd.DataFrame(inc), "balance": pd.DataFrame(bal),
            "cashflow": pd.DataFrame(cf)}


def _loader(tickers=("AAPL",)):
    return lambda **kwargs: _frames(tickers)


def test_field_map_covers_every_quality_field():
    # Drift guard: SimFin must supply exactly the fields the quality signal reads,
    # so repointing the source needs no change to the Piotroski math.
    assert set(FIELD_MAP) == set(FIELDS)


def test_publish_date_becomes_knowledge_date():
    records, quarantine, _ = records_from_frames(_frames())
    assert not records.empty
    assert quarantine == []
    latest = records[(records.field == "revenue")
                     & (records.event_date == pd.Timestamp("2020-12-31"))]
    assert len(latest) == 1
    row = latest.iloc[0]
    # knowledge_date is the Publish Date, NOT the fiscal period end
    assert row.knowledge_date == pd.Timestamp("2021-02-15")
    assert row.event_date == pd.Timestamp("2020-12-31")
    assert row.value == 1000.0
    assert row.source == SOURCE


def test_no_peek_figure_published_after_as_of_is_invisible(tmp_path):
    # EVAL (a): the Q4 figure (ends 2020-12-31, published 2021-02-15) must not be
    # visible on a decision date between those two dates.
    store = PITStore(tmp_path / "s.sqlite")
    load_simfin_fundamentals(store, frames_loader=_loader(), tickers=["AAPL"])

    before = store.get_data("revenue", "AAPL", date(2021, 1, 15))
    assert pd.Timestamp("2020-12-31") not in set(before["event_date"])  # not yet public
    assert pd.Timestamp("2020-09-30") in set(before["event_date"])      # prior IS public

    after = store.get_data("revenue", "AAPL", date(2021, 3, 1))
    assert pd.Timestamp("2020-12-31") in set(after["event_date"])       # now public


def test_quality_populates_for_large_cap_once_loaded(tmp_path):
    # EVAL (b): with SimFin loaded, quality is scoreable (not n/a) for known names.
    store = PITStore(tmp_path / "s.sqlite")
    load_simfin_fundamentals(store, frames_loader=_loader(("AAPL", "LLY")),
                             tickers=["AAPL", "LLY"])
    for tkr in ("AAPL", "LLY"):
        q = quality_score(tkr, date(2021, 3, 1), store=store)
        assert not q.insufficient_data, q.reason
        assert q.score is not None and 0.0 <= q.score <= 100.0
        assert q.f_score is not None


def test_ticker_absent_from_simfin_is_quarantined_not_fatal(tmp_path):
    store = PITStore(tmp_path / "s.sqlite")
    res = load_simfin_fundamentals(store, frames_loader=_loader(("AAPL",)),
                                   tickers=["AAPL", "ZZZZ"])
    by = {(q["ticker"], q["vendor"]) for q in res.quarantine}
    assert ("ZZZZ", SOURCE) in by
    assert ("AAPL", SOURCE) not in by
    # the present ticker still loaded cleanly
    assert not quality_score("AAPL", date(2021, 3, 1), store=store).insufficient_data


def test_unusable_fact_is_skipped_not_written():
    frames = _frames(("AAPL",))
    frames["income"].loc[0, "Net Income"] = None  # one fact with no value
    records, _, n_skipped = records_from_frames(frames)
    assert n_skipped >= 1
    ni = records[records.field == "net_income"]
    assert pd.Timestamp("2020-09-30") not in set(ni.event_date)  # NaN not written
