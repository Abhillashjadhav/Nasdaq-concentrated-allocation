"""Momentum signal tests (ARCHITECTURE.md §4, §8, §9.5).

Golden case: a deterministic linear price ramp lets every 200-day window mean be
exactly (first+last)/2, so the expected sub-scores and final score are derived
independently from the implementation and asserted decimal-exact. Plus an
insufficient-history guard test and a no-future-leak test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from signals.momentum import (
    HIGH_52W_WINDOW,
    MIN_HISTORY,
    MOM_BAND,
    PROX_BAND,
    SLOPE_BAND,
    TREND_BAND,
    WEIGHTS,
    momentum_score,
)
from store.store import PITStore

N = 300
P0, STEP = 100.0, 0.05  # chronological price ramp: p_k = 100 + 0.05*k
DATES = pd.bdate_range("2018-01-01", periods=N)
PRICES = np.array([P0 + STEP * k for k in range(N)])  # index 0 = oldest


def _ramp_store(tmp_path, n=N) -> PITStore:
    s = PITStore(tmp_path / "mom.sqlite")
    s.put_data(
        pd.DataFrame(
            {
                "ticker": "RAMP", "field": "close", "value": PRICES[:n],
                "event_date": DATES[:n], "knowledge_date": DATES[:n],
                "source": "fixture",
            }
        )
    )
    return s


def _band(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / (hi - lo))) * 100.0


def _expected_score(p: np.ndarray) -> dict:
    """Independent re-derivation of the documented math (newest-first offsets)."""
    last = len(p) - 1

    def chron(off):  # value `off` trading days before the latest
        return p[last - off]

    mom = chron(21) / chron(252) - 1.0
    # linear ramp => window mean == (first + last) / 2
    sma_now = (p[last - 199] + p[last]) / 2          # mean of last 200
    sma_prev = (p[last - 21 - 199] + p[last - 21]) / 2  # mean of last 200, 21d ago
    trend = chron(0) / sma_now - 1.0
    slope = sma_now / sma_prev - 1.0
    # monotonic ramp => max over the last 252 is one of the window endpoints
    high = max(p[last - (HIGH_52W_WINDOW - 1)], p[last])
    prox = chron(0) / high - 1.0
    raw = {"mom_12_1": mom, "price_vs_sma200": trend, "sma200_slope": slope,
           "high_52w_proximity": prox}
    sub = {
        "mom_12_1": _band(mom, *MOM_BAND),
        "price_vs_sma200": _band(trend, *TREND_BAND),
        "sma200_slope": _band(slope, *SLOPE_BAND),
        "high_52w_proximity": _band(prox, *PROX_BAND),
    }
    return {"raw": raw, "score": sum(WEIGHTS[k] * sub[k] for k in WEIGHTS), "sub": sub}


def test_golden_case(tmp_path):
    store = _ramp_store(tmp_path)
    res = momentum_score("RAMP", DATES[-1], store=store)
    exp = _expected_score(PRICES)

    assert res.insufficient_data is False
    assert res.score == pytest.approx(exp["score"], abs=1e-9)
    for k in WEIGHTS:
        assert res.components["subscores"][k] == pytest.approx(exp["sub"][k], abs=1e-9)
        assert res.components["raw"][k] == pytest.approx(exp["raw"][k], abs=1e-12)


def test_golden_case_descending_exercises_proximity(tmp_path):
    """A descending ramp keeps every window mean = (first+last)/2 but puts the
    current price BELOW the 52-week high, so the proximity sub-score lands in its
    interior (not the at-the-high boundary the ascending ramp produces)."""
    prices = np.array([200.0 - 0.05 * k for k in range(N)])  # chronological, falling
    s = PITStore(tmp_path / "down.sqlite")
    s.put_data(
        pd.DataFrame(
            {"ticker": "DOWN", "field": "close", "value": prices,
             "event_date": DATES, "knowledge_date": DATES, "source": "fixture"}
        )
    )
    res = momentum_score("DOWN", DATES[-1], store=s)
    exp = _expected_score(prices)

    assert res.score == pytest.approx(exp["score"], abs=1e-9)
    for k in WEIGHTS:
        assert res.components["subscores"][k] == pytest.approx(exp["sub"][k], abs=1e-9)
    # every sub-score is strictly interior here -> real band arithmetic, no clamps
    assert all(0.0 < v < 100.0 for v in res.components["subscores"].values())


def test_insufficient_history(tmp_path):
    store = _ramp_store(tmp_path, n=MIN_HISTORY - 1)  # one short of the requirement
    res = momentum_score("RAMP", DATES[MIN_HISTORY - 2], store=store)
    assert res.insufficient_data is True
    assert res.score is None
    assert "insufficient_history" in res.reason
    # never silently 0 or 50
    assert res.score not in (0, 50)


def test_no_future_leak(tmp_path):
    """A score as of T must not depend on rows known after T."""
    full = _ramp_store(tmp_path)  # all 300 rows
    as_of = DATES[260]  # 261 rows visible (>= MIN_HISTORY)

    from_full = momentum_score("RAMP", as_of, store=full).score
    partial = _ramp_store(tmp_path / "p", n=261)  # only rows 0..260 exist
    from_partial = momentum_score("RAMP", as_of, store=partial).score

    assert from_full == pytest.approx(from_partial, abs=1e-12)
    assert_no_future_rows(full, ["close"], ["RAMP"], [DATES[260], DATES[280], DATES[-1]])
