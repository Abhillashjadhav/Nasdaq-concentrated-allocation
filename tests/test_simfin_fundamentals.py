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


# --- BUG: SimFin free bulk omits Publish Date -> every fact was dropped at the PIT
#     gate, leaving quality n/a for the whole universe. Fall back to a no-peek-safe
#     knowledge_date (period end + filing lag) instead of discarding the fact. ------

def _frames_no_publish(ticker="LLY"):
    """Two annual periods with Publish Date MISSING (free-tier shape) + Fiscal Period."""
    def inc(report, rev, gp, ni, sh):
        return {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": report,
                "Publish Date": None, "Revenue": rev, "Gross Profit": gp,
                "Net Income": ni, "Shares (Diluted)": sh}

    def bal(report, ta, ca, cl, ltd):
        return {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": report,
                "Publish Date": None, "Total Assets": ta, "Total Current Assets": ca,
                "Total Current Liabilities": cl, "Long Term Debt": ltd}

    def cf(report, cfo):
        return {"Ticker": ticker, "Fiscal Period": "FY", "Report Date": report,
                "Publish Date": None, "Net Cash from Operating Activities": cfo}

    return {
        "income": pd.DataFrame([inc("2022-12-31", 900, 400, 100, 170),
                                inc("2023-12-31", 1000, 480, 150, 160)]),
        "balance": pd.DataFrame([bal("2022-12-31", 1000, 450, 200, 350),
                                 bal("2023-12-31", 1100, 500, 200, 300)]),
        "cashflow": pd.DataFrame([cf("2022-12-31", 180), cf("2023-12-31", 200)]),
    }


def test_quality_populates_when_publish_date_missing(tmp_path):
    # The regression: with Publish Date empty the old code skipped every fact and
    # quality came back n/a. The fallback keeps the facts, so quality scores.
    store = PITStore(tmp_path / "s.sqlite")
    load_simfin_fundamentals(store, frames_loader=lambda **k: _frames_no_publish("LLY"),
                             tickers=["LLY"])
    q = quality_score("LLY", date(2024, 6, 30), store=store)
    assert not q.insufficient_data, q.reason
    assert q.score is not None and q.f_score is not None


def test_publish_date_fallback_is_no_peek_safe(tmp_path):
    # The derived knowledge_date is period end + lag — strictly AFTER the period end
    # — so an annual figure is invisible right after year-end and visible only later.
    records, _, _ = records_from_frames(_frames_no_publish("LLY"))
    fy23 = records[(records.field == "revenue")
                   & (records.event_date == pd.Timestamp("2023-12-31"))].iloc[0]
    assert fy23.knowledge_date == pd.Timestamp("2023-12-31") + pd.Timedelta(days=90)

    store = PITStore(tmp_path / "s.sqlite")
    store.put_data(records)
    just_after = store.get_data("revenue", "LLY", date(2024, 1, 15))
    assert pd.Timestamp("2023-12-31") not in set(just_after["event_date"])  # not yet "known"
    later = store.get_data("revenue", "LLY", date(2024, 6, 30))
    assert pd.Timestamp("2023-12-31") in set(later["event_date"])           # known after lag


def test_present_publish_date_still_wins():
    # The fallback fires ONLY on a missing date; a real Publish Date is untouched.
    records, _, _ = records_from_frames(_frames())  # _frames carries real publish dates
    row = records[(records.field == "revenue")
                  & (records.event_date == pd.Timestamp("2020-12-31"))].iloc[0]
    assert row.knowledge_date == pd.Timestamp("2021-02-15")  # the actual Publish Date


# --- PERFORMANCE: the bulk CSVs carry ~7,000 companies. A small run must filter to
#     its target tickers at LOAD time, not iterate the whole dataset (which hung for
#     tens of minutes on a single name). The ticker filter is pushed into the loader. -

def test_tickers_are_pushed_into_the_loader(tmp_path):
    seen = {}

    def _spy_loader(**kwargs):
        seen.update(kwargs)            # capture what load_simfin_fundamentals forwards
        return _frames(("AAPL",))

    store = PITStore(tmp_path / "s.sqlite")
    load_simfin_fundamentals(store, frames_loader=_spy_loader, tickers=["AAPL", "LLY"])
    # the loader receives the ticker list so it can cut each CSV before any iteration
    assert seen.get("tickers") == ["AAPL", "LLY"]


def test_download_frames_filters_each_csv_to_target_tickers(monkeypatch):
    # _download_frames must filter each CSV to the requested tickers at load time
    # (before concat) so the 7,000-row bulk CSV doesn't bloat iteration for a tiny run.
    import data.simfin_client as sc_mod

    big = pd.DataFrame([
        {"Ticker": "AAPL", "Report Date": "2023-12-31"},
        {"Ticker": "MSFT", "Report Date": "2023-12-31"},
        {"Ticker": "LLY",  "Report Date": "2023-12-31"},
    ])

    monkeypatch.setattr(sc_mod, "_read_csv", lambda *a, **k: big.copy())
    monkeypatch.setattr(sc_mod, "_ensure_dataset", lambda *a, **k: None)

    frames = sc_mod._download_frames(api_key="x", cache_dir="/tmp/_sf_perf",
                                     refresh=False, variants=("annual",),
                                     tickers=["AAPL"])
    for name, df in frames.items():
        assert set(df["Ticker"]) == {"AAPL"}, name   # MSFT/LLY dropped at load time


# --- PRODUCTION MISS: SimFin's free bulk ships Publish Date as an EMPTY STRING, not
#     NaT. pd.isna("") is False, so the PR #33 fallback never fired on the real ingest
#     path: pd.Timestamp("") -> NaT -> rows rejected by the not-null schema -> quality
#     n/a in production while the NaT fixture passed. Guard with real-shaped data. -----

def _frames_blank_publish(ticker="LLY"):
    """Free-tier shape: Publish Date present but blank ('' empty string)."""
    frames = _frames_no_publish(ticker)
    for df in frames.values():
        df["Publish Date"] = ""  # exactly what SimFin's free bulk ships
    return frames


def test_blank_string_publish_date_is_recovered_on_real_path(tmp_path):
    frames = _frames_blank_publish("LLY")
    records, _, n_skipped = records_from_frames(frames, tickers=["LLY"])
    assert not records.empty and n_skipped == 0
    assert records["knowledge_date"].notna().all()                    # no NaT slipped through
    assert (records["knowledge_date"] > records["event_date"]).all()  # no-peek safe (lagged)

    store = PITStore(tmp_path / "s.sqlite")
    store.put_data(records)                                           # raised SchemaErrors pre-fix
    q = quality_score("LLY", date(2024, 6, 30), store=store)
    assert not q.insufficient_data, q.reason
    assert q.score is not None


def test_coerce_date_treats_blank_and_unparseable_as_missing():
    from data.simfin_client import _coerce_date
    for missing in ("", "   ", None, pd.NaT, float("nan"), "not-a-date"):
        assert _coerce_date(missing) is None, missing
    assert _coerce_date("2023-02-20") == pd.Timestamp("2023-02-20")
    assert _coerce_date(pd.Timestamp("2023-02-20")) == pd.Timestamp("2023-02-20")


def test_filing_lag_by_period():
    from data.simfin_client import _filing_lag
    assert _filing_lag("FY") == pd.Timedelta(days=90)
    assert _filing_lag("Q3") == pd.Timedelta(days=45)
    assert _filing_lag(None) == pd.Timedelta(days=75)
    assert _filing_lag(float("nan")) == pd.Timedelta(days=75)


def test_diagnostic_identifies_failure_modes():
    # The diagnostic distinguishes the four failure modes on synthetic inputs, so it
    # pinpoints (not guesses) where the SimFin->quality chain breaks on a real cache.
    from diagnose_simfin_quality import _fixture_frames, diagnose

    # publish-date skip is now fixed -> the free-tier shape scores cleanly
    assert diagnose(_fixture_frames("LLY", publish_empty=True), "LLY") == "ok"
    # a name absent from the frames -> ticker-key mismatch
    assert diagnose(_fixture_frames("LLY"), "ZZZZ") == "ticker-key mismatch"
    # a missing required column -> column-name mismatch
    broken = _fixture_frames("LLY")
    broken["balance"] = broken["balance"].drop(columns=["Total Assets"])
    assert diagnose(broken, "LLY") == "column-name mismatch"
