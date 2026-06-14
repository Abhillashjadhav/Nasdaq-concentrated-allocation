"""Macro regime-gate tests (ARCHITECTURE.md §4, §6, §8, §9.9).

Proves the AND gate: risk-off only when HY-OAS widening, Fed tightening, and a VIX
break all coincide. A single condition keeps the gate OPEN (AND, not OR). Plus a
release-lag no-leak case and a missing-series fail-safe-to-risk-off case.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from evals.no_peek import assert_no_future_rows
from macro.regime import (
    HY_OAS,
    MACRO_TICKER,
    POLICY_RATE,
    VIX,
    regime_state,
)
from store.store import PITStore

FIELDS = [HY_OAS, POLICY_RATE, VIX]
AS_OF = date(2020, 3, 31)
OLD = "2020-01-02"
OLD_REL = "2020-01-03"
RECENT = "2020-03-30"
RECENT_REL = "2020-03-31"


def _pt(field, event_date, release_date, value) -> dict:
    return {"ticker": MACRO_TICKER, "field": field, "value": float(value),
            "event_date": pd.Timestamp(event_date),
            "knowledge_date": pd.Timestamp(release_date), "source": "fixture"}


def _store(tmp_path, rows) -> PITStore:
    s = PITStore(tmp_path / "macro.sqlite")
    s.put_data(pd.DataFrame(rows))
    return s


def test_all_conditions_coincide_gate_closed(tmp_path):
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, RECENT_REL, 8.0),     # +4.0pp
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, RECENT_REL, 3.5),  # +1.0pp
        _pt(VIX, OLD, OLD_REL, 15.0), _pt(VIX, RECENT, RECENT_REL, 60.0),         # >=30
    ]
    res = regime_state(AS_OF, store=_store(tmp_path, rows))
    assert res.insufficient_data is False
    assert res.state == "risk_off"
    assert res.gate_closed is True
    assert all(res.conditions.values())


def test_benign_macro_gate_open(tmp_path):
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, RECENT_REL, 4.2),
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, RECENT_REL, 2.5),
        _pt(VIX, OLD, OLD_REL, 15.0), _pt(VIX, RECENT, RECENT_REL, 15.0),
    ]
    res = regime_state(AS_OF, store=_store(tmp_path, rows))
    assert res.state == "risk_on"
    assert res.gate_closed is False
    assert not any(res.conditions.values())


def test_single_condition_keeps_gate_open(tmp_path):
    """Only VIX breaks; OAS and rates are calm -> AND gate stays OPEN (not OR)."""
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, RECENT_REL, 4.1),  # +0.1pp
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, RECENT_REL, 2.5),
        _pt(VIX, OLD, OLD_REL, 15.0), _pt(VIX, RECENT, RECENT_REL, 60.0),       # break
    ]
    res = regime_state(AS_OF, store=_store(tmp_path, rows))
    assert res.state == "risk_on"
    assert res.gate_closed is False
    assert res.conditions["vix_breaking"] is True
    assert sum(bool(v) for v in res.conditions.values()) == 1  # exactly one fired


def test_vix_level_break_without_lookback(tmp_path):
    """VIX has only a single (recent) point -> no lookback, but the absolute-level
    trigger (>=30) still fires; with OAS/rate also firing the gate closes."""
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, RECENT_REL, 8.0),
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, RECENT_REL, 3.5),
        _pt(VIX, RECENT, RECENT_REL, 60.0),  # single point -> no lookback
    ]
    res = regime_state(AS_OF, store=_store(tmp_path, rows))
    assert res.insufficient_data is False
    assert res.conditions["vix_breaking"] is True
    assert res.components["vix_change"] is None  # confirms the no-lookback path
    assert res.state == "risk_off" and res.gate_closed is True


def test_missing_series_fails_safe_to_risk_off(tmp_path):
    """A missing macro series must NOT default to risk-on (stay invested) — it
    fails safe to risk-off with the gate closed and the gap named."""
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, RECENT_REL, 8.0),
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, RECENT_REL, 3.5),
        # VIX deliberately absent
    ]
    res = regime_state(AS_OF, store=_store(tmp_path, rows))
    assert res.insufficient_data is True
    assert res.gate_closed is True            # safe default = defensive
    assert res.state == "risk_off"
    assert any(m.startswith(VIX) for m in res.missing)


def test_no_future_leak_respects_release_lag(tmp_path):
    """A risk-off spike released after as_of must not flip the gate early."""
    rows = [
        _pt(HY_OAS, OLD, OLD_REL, 4.0), _pt(HY_OAS, RECENT, "2020-04-05", 8.0),
        _pt(POLICY_RATE, OLD, OLD_REL, 2.5), _pt(POLICY_RATE, RECENT, "2020-04-05", 3.5),
        _pt(VIX, OLD, OLD_REL, 15.0), _pt(VIX, RECENT, "2020-04-05", 60.0),
    ]
    store = _store(tmp_path, rows)

    pre = regime_state(date(2020, 3, 31), store=store)   # spike not yet released
    assert pre.state == "risk_on" and pre.gate_closed is False

    post = regime_state(date(2020, 4, 6), store=store)   # spike now public
    assert post.state == "risk_off" and post.gate_closed is True

    assert_no_future_rows(store, FIELDS, [MACRO_TICKER],
                          [date(2020, 3, 31), date(2020, 4, 6), date(2020, 5, 1)])
