"""Rank-as-of-date funnel eval (ARCHITECTURE.md §2, §6, §8).

Rank-ordering golden case: known sub-scores -> known descending order + correct
percentile; an unrankable name (no scorable signal) is dropped. Plus the rendered
markdown's honesty framing and an offline run_ranking over a cached SIC universe.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from report.ranking import rank_as_of, render_ranking_markdown
from run import RunConfig, run_ranking
from store.schema import COLUMNS
from store.store import PITStore


def _scorers(table):
    def make(sig):
        return lambda t, as_of, store: table.get(t, {}).get(sig)
    return {"alpha": make("alpha"), "beta": make("beta")}


def test_rank_ordering_golden():
    # composites: AAA=85, CCC=80, BBB=50 -> AAA, CCC, BBB; DDD unscored -> dropped
    table = {"AAA": {"alpha": 90, "beta": 80},
             "BBB": {"alpha": 50, "beta": 50},
             "CCC": {"alpha": 70, "beta": 90},
             "DDD": {}}
    res = rank_as_of(date(2020, 1, 1), ["AAA", "BBB", "CCC", "DDD"], store=None,
                     scorers=_scorers(table), top_n=10, with_forward_return=False)

    assert [r.ticker for r in res.rows] == ["AAA", "CCC", "BBB"]   # DDD dropped
    assert res.n_scored == 3
    assert res.rows[0].composite == 85.0 and res.rows[0].rank == 1
    assert res.rows[0].percentile == 100.0                         # top of the universe
    assert res.rows[-1].percentile == pytest.approx(100.0 / 3)     # bottom
    assert res.rows[0].subscores == {"alpha": 90, "beta": 80}


def test_top_n_truncates():
    table = {t: {"alpha": i} for i, t in enumerate(["A", "B", "C", "D", "E"])}
    res = rank_as_of(date(2020, 1, 1), list(table), store=None,
                     scorers=_scorers(table), top_n=2, with_forward_return=False)
    assert len(res.rows) == 2 and res.n_scored == 5


def test_render_markdown_states_it_is_not_a_win_probability():
    table = {"AAA": {"alpha": 90, "beta": 80}}
    res = rank_as_of(date(2020, 1, 1), ["AAA"], store=None,
                     scorers=_scorers(table), with_forward_return=False)
    md = render_ranking_markdown(
        [res], coverage={"n_universe": 100, "n_classified": 50, "n_priced": 40, "n_quarantined": 5},
        signal_names=["alpha", "beta"])
    assert "not a win-probability" in md
    assert "SURVIVORSHIP NOTE" in md and "SURVIVOR-LIMITED" in md
    assert "quarantined: 5" in md
    assert "as-of 2020-01-01" in md and "AAA" in md


def _sector(ticker, code, filed="2015-03-01"):
    d = pd.Timestamp(filed)
    return {"ticker": ticker, "field": "sector", "value": float(code),
            "event_date": d, "knowledge_date": d, "source": "simfin"}


def test_run_ranking_offline(tmp_path):
    store = PITStore(tmp_path / "rank.sqlite")
    store.put_data(pd.DataFrame([_sector("TECHX", 1.0), _sector("HEALX", 2.0),
                                 _sector("BANKX", 9.0)], columns=COLUMNS))
    config = RunConfig(store=store, tickers=[], entry_dates=[date(2020, 1, 1)],
                       active_signals=["x"], output_dir=str(tmp_path / "out"))
    scorers = {"x": lambda t, as_of, st: {"TECHX": 90.0, "HEALX": 70.0}.get(t)}

    rr = run_ranking(config, asof_dates=[date(2020, 1, 1)], top_n=10,
                     symbols=["TECHX", "HEALX", "BANKX"], n_quarantined=3, scorers=scorers)

    assert rr.coverage["n_classified"] == 2           # BANKX excluded (out-of-scope sector)
    assert rr.coverage["n_quarantined"] == 3
    assert [r.ticker for r in rr.results[0].rows] == ["TECHX", "HEALX"]
    assert Path(rr.report_path).exists()
    assert "not a win-probability" in Path(rr.report_path).read_text()


