"""Quality / profitability tests (ARCHITECTURE.md §4, §6, §8, §9.8).

Golden case: two periods of known fundamentals -> an integer-exact F-score with a
deliberate 5-of-9 pass/fail mix (every check individually verified) and a
decimal-exact combined sub-score. Plus insufficient-history, no-coverage,
invalid-denominator, and the all-important filing-lag no-leak case.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from signals.quality import FIELDS, GP_BAND, WEIGHTS, quality_score
from store.store import PITStore

PRIOR = {  # period end 2019-12-31, filed 2020-02-28
    "net_income": 100, "cfo": 120, "total_assets": 1000, "long_term_debt": 200,
    "current_assets": 400, "current_liabilities": 200, "shares_outstanding": 50,
    "gross_profit": 400, "revenue": 800,
}
LATEST = {  # period end 2020-12-31, filed 2021-02-26
    "net_income": 90, "cfo": 150, "total_assets": 1100, "long_term_debt": 300,
    "current_assets": 550, "current_liabilities": 200, "shares_outstanding": 60,
    "gross_profit": 480, "revenue": 1000,
}

EXPECTED_CHECKS = {
    "ni_positive": True,            # 90 > 0
    "cfo_positive": True,           # 150 > 0
    "roa_rising": False,            # 90/1100=0.0818 < 100/1000=0.10
    "accrual_cfo_gt_ni": True,      # 150 > 90
    "ltd_ratio_falling": False,     # 300/1100=0.273 > 200/1000=0.20
    "current_ratio_rising": True,   # 550/200=2.75 > 400/200=2.0
    "no_new_shares": False,         # 60 > 50 (shares issued)
    "gross_margin_rising": False,   # 480/1000=0.48 < 400/800=0.50
    "asset_turnover_rising": True,  # 1000/1100=0.909 > 800/1000=0.80
}
EXPECTED_F = 5  # five True above


def _period(ticker, period_end, filing_date, vals) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": ticker, "field": f, "value": float(v),
          "event_date": pd.Timestamp(period_end),
          "knowledge_date": pd.Timestamp(filing_date), "source": "fixture"}
         for f, v in vals.items()]
    )


def _two_period_store(tmp_path, ticker="Q") -> PITStore:
    s = PITStore(tmp_path / "fund.sqlite")
    s.put_data(
        pd.concat(
            [
                _period(ticker, "2019-12-31", "2020-02-28", PRIOR),
                _period(ticker, "2020-12-31", "2021-02-26", LATEST),
            ],
            ignore_index=True,
        )
    )
    return s


def _expected_score():
    gp = LATEST["gross_profit"] / LATEST["total_assets"]
    f_comp = EXPECTED_F / 9.0 * 100.0
    gp_comp = min(1.0, max(0.0, (gp - GP_BAND[0]) / (GP_BAND[1] - GP_BAND[0]))) * 100.0
    return WEIGHTS["f_score"] * f_comp + WEIGHTS["gross_profitability"] * gp_comp, gp


def test_golden_case(tmp_path):
    res = quality_score("Q", date(2021, 3, 31), store=_two_period_store(tmp_path))
    assert res.insufficient_data is False
    assert res.f_score == EXPECTED_F  # integer-exact
    for name, expected in EXPECTED_CHECKS.items():
        assert res.components["checks"][name] is expected, name

    expected_score, gp = _expected_score()
    assert res.components["gross_profitability"] == pytest.approx(gp, abs=1e-12)
    assert res.score == pytest.approx(expected_score, abs=1e-9)


def test_insufficient_history(tmp_path):
    s = PITStore(tmp_path / "one.sqlite")
    s.put_data(_period("Q", "2019-12-31", "2020-02-28", PRIOR))  # only one period
    res = quality_score("Q", date(2020, 6, 30), store=s)
    assert res.insufficient_data is True
    assert res.score is None and res.f_score is None
    assert res.reason == "insufficient_fundamental_history"


def test_no_coverage(tmp_path):
    s = PITStore(tmp_path / "empty.sqlite")
    s.put_data(_period("OTHER", "2019-12-31", "2020-02-28", PRIOR))
    res = quality_score("Q", date(2021, 3, 31), store=s)
    assert res.insufficient_data is True
    assert res.reason == "no_fundamental_coverage"


def test_invalid_denominator_flagged(tmp_path):
    bad = dict(LATEST, total_assets=0)  # zero denominator must fail loud
    s = PITStore(tmp_path / "bad.sqlite")
    s.put_data(
        pd.concat(
            [_period("Q", "2019-12-31", "2020-02-28", PRIOR),
             _period("Q", "2020-12-31", "2021-02-26", bad)],
            ignore_index=True,
        )
    )
    res = quality_score("Q", date(2021, 3, 31), store=s)
    assert res.insufficient_data is True
    assert res.reason == "invalid_fundamentals"


def test_filing_lag_no_leak(tmp_path):
    """The latest period ends 2020-12-31 but is FILED 2021-02-26: it must be
    invisible until filed, so before that only one period is visible (cannot score)."""
    store = _two_period_store(tmp_path)

    before_filing = quality_score("Q", date(2021, 1, 31), store=store)
    assert before_filing.insufficient_data is True  # Q4 not yet filed -> only 1 period
    assert before_filing.reason == "insufficient_fundamental_history"

    on_filing = quality_score("Q", date(2021, 2, 26), store=store)
    assert on_filing.insufficient_data is False
    assert on_filing.f_score == EXPECTED_F

    assert_no_future_rows(store, FIELDS, ["Q"],
                          [date(2021, 1, 31), date(2021, 2, 26), date(2021, 4, 1)])
