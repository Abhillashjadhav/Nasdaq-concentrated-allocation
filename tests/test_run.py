"""End-to-end orchestrator tests (ARCHITECTURE.md §5, §7, §9).

Integration: a full run on a small synthetic store -> a complete report + verdict.
Partial readiness: only quality active -> others marked not_run, no crash. Fail
loud: a missing required input -> clear error, no fabricated report. Plus the
default-registry adapter-availability contract.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtest.labels import BENCHMARK_TICKER
from run import DEFAULT_SIGNALS, PipelineError, RunConfig, SignalSpec, run

YEARS = (2016, 2017, 2018)


def _price(ticker, event_date, value):
    return {"ticker": ticker, "field": "close", "value": float(value),
            "event_date": pd.Timestamp(event_date),
            "knowledge_date": pd.Timestamp(event_date), "source": "fixture"}


def _build_store(tmp_path, *, with_benchmark=True, n_win=20, n_lose=20):
    """Continuous compounding paths (one year-end close each) so entry year Y
    reads Dec-31(Y-1) and its exit reads Dec-31(Y) with no cross-year overlap.
    Winners compound +30%/yr, benchmark +10%/yr (-> +20pp excess, winner), losers
    +5%/yr (-> -5pp, not winner)."""
    from store.store import PITStore
    s = PITStore(tmp_path / "run.sqlite")
    winners = [f"W{i}" for i in range(n_win)]
    losers = [f"L{i}" for i in range(n_lose)]
    cal = range(min(YEARS) - 1, max(YEARS) + 1)  # year-ends needed: entry(Y-1)..exit(maxY)

    def _path(tickers, rate):
        out = []
        for t in tickers:
            for y in cal:
                out.append(_price(t, f"{y}-12-31", 100.0 * (1.0 + rate) ** (y - (min(YEARS) - 1))))
        return out

    rows = _path(winners, 0.30) + _path(losers, 0.05)
    if with_benchmark:
        rows += _path([BENCHMARK_TICKER], 0.10)
    s.put_data(pd.DataFrame(rows))
    return s, winners, losers


def _planted_scorer(winners):
    wset = set(winners)
    return lambda ticker, as_of, store: (90.0 if ticker in wset else 30.0, False)


def _never(*_a, **_k):  # a scorer that must never be called (for not_run signals)
    raise AssertionError("scorer for a not_run signal was called")


def _config(tmp_path, store, winners, losers, **over):
    base = dict(
        store=store, tickers=winners + losers,
        entry_dates=[date(y, 1, 1) for y in YEARS],
        active_signals=["alpha", "beta"], output_dir=str(tmp_path / "out"),
        signals={"alpha": SignalSpec("alpha", _planted_scorer(winners), True),
                 "beta": SignalSpec("beta", _planted_scorer(winners), True)},
        min_samples_per_arm=10, two_arm_kwargs={"threshold": 60.0, "n_boot": 200},
    )
    base.update(over)
    return RunConfig(**base)


def test_end_to_end_produces_report_with_verdict(tmp_path):
    store, winners, losers = _build_store(tmp_path)
    res = run(_config(tmp_path, store, winners, losers))

    assert res.report.verdict in {"GO", "MARGINAL", "KILL"}
    assert res.report.verdict == "GO"  # two planted-edge archetypes confirm
    assert res.statuses == {"alpha": "evaluated", "beta": "evaluated"}
    report_file = Path(res.report_path)
    assert report_file.exists()
    assert "## VERDICT: GO" in report_file.read_text()


def test_partial_readiness_marks_others_not_run(tmp_path):
    store, winners, losers = _build_store(tmp_path)
    cfg = _config(
        tmp_path, store, winners, losers,
        active_signals=["quality", "revisions", "insiders"],
        signals={
            "quality": SignalSpec("quality", _planted_scorer(winners), True),
            "revisions": SignalSpec("revisions", _never, False),
            "insiders": SignalSpec("insiders", _never, False),
        },
    )
    res = run(cfg)

    assert res.statuses["quality"] == "evaluated"
    assert res.statuses["revisions"] == "not_run"
    assert res.statuses["insiders"] == "not_run"
    assert set(res.not_run) == {"revisions", "insiders"}
    assert res.report.verdict in {"GO", "MARGINAL", "KILL"}  # ran quality, didn't crash
    assert Path(res.report_path).exists()


def test_mixed_readiness_distinguishes_all_statuses(tmp_path):
    """evaluated vs insufficient_data (adapter present, data missing) vs not_run
    (no adapter) vs unknown_signal must all coexist without crashing."""
    store, winners, losers = _build_store(tmp_path)
    cfg = RunConfig(
        store=store, tickers=winners + losers,
        entry_dates=[date(y, 1, 1) for y in YEARS],
        active_signals=["alpha", "broken", "norun", "ghost"],  # ghost: not in registry
        output_dir=str(tmp_path / "mixed"),
        signals={
            "alpha": SignalSpec("alpha", _planted_scorer(winners), True),
            "broken": SignalSpec("broken", lambda *_: (None, True), True),  # always insufficient
            "norun": SignalSpec("norun", _never, False),
        },
        min_samples_per_arm=10, two_arm_kwargs={"threshold": 60.0, "n_boot": 200},
    )
    res = run(cfg)

    assert res.statuses["alpha"] == "evaluated"
    assert res.statuses["broken"].startswith("insufficient_data")  # adapter ok, data missing
    assert res.statuses["norun"] == "not_run"                      # no adapter at all
    assert res.statuses["ghost"] == "unknown_signal"
    assert res.not_run == ["norun"]
    assert Path(res.report_path).exists()  # still produced a report from the one good signal


def test_fail_loud_when_required_input_missing(tmp_path):
    # no benchmark series -> every label is not-yet-known -> nothing evaluable
    store, winners, losers = _build_store(tmp_path, with_benchmark=False)
    cfg = _config(tmp_path, store, winners, losers)
    with pytest.raises(PipelineError):
        run(cfg)
    assert not (Path(cfg.output_dir) / "report.md").exists()  # no fabricated output


def test_fail_loud_when_no_runnable_signal(tmp_path):
    store, winners, losers = _build_store(tmp_path)
    # both active signals lack a live adapter (default registry) -> refuse to run
    cfg = RunConfig(
        store=store, tickers=winners + losers,
        entry_dates=[date(y, 1, 1) for y in YEARS],
        active_signals=["revisions", "insiders"], output_dir=str(tmp_path / "out2"),
    )
    with pytest.raises(PipelineError):
        run(cfg)


def test_default_registry_adapter_availability():
    assert DEFAULT_SIGNALS["momentum"].adapter_available is True
    assert DEFAULT_SIGNALS["quality"].adapter_available is True
    assert DEFAULT_SIGNALS["revisions"].adapter_available is False
    assert DEFAULT_SIGNALS["insiders"].adapter_available is True  # Form 4 adapter (data/form4.py)


def test_cli_has_min_obs_per_slice_flag():
    from run import _build_parser
    args = _build_parser().parse_args(["--db", "x", "--tickers", "A", "--min-obs-per-slice", "5"])
    assert args.min_obs_per_slice == 5
    assert _build_parser().parse_args(["--db", "x", "--tickers", "A"]).min_obs_per_slice is None


def _sic_row(ticker, sic, filed="2015-03-01"):
    d = pd.Timestamp(filed)
    return {"ticker": ticker, "field": "sic", "value": float(sic),
            "event_date": d, "knowledge_date": d, "source": "edgar"}


def test_resolve_universe_candidates_keeps_only_hc_tech(tmp_path):
    # GO/KILL on the real universe: classified tech+health in, others out,
    # resolved point-in-time from the cached SIC store.
    from store.schema import COLUMNS
    from store.store import PITStore

    from run import resolve_universe_candidates

    store = PITStore(tmp_path / "uni.sqlite")
    store.put_data(pd.DataFrame(
        [_sic_row("TECHX", 7372), _sic_row("HEALX", 2836), _sic_row("BANKX", 6022)],
        columns=COLUMNS))
    got = resolve_universe_candidates(["TECHX", "HEALX", "BANKX"], store, date(2020, 1, 1))
    assert set(got) == {"TECHX", "HEALX"}  # BANKX (financial SIC) excluded


def test_resolve_universe_candidates_empty_fails_loud(tmp_path):
    from store.store import PITStore

    from run import resolve_universe_candidates

    store = PITStore(tmp_path / "empty.sqlite")  # no SIC cached
    with pytest.raises(PipelineError):
        resolve_universe_candidates(["AAA", "BBB"], store, date(2020, 1, 1))


def test_cli_universe_mode_needs_no_tickers():
    from run import _build_parser
    args = _build_parser().parse_args(["--db", "x", "--universe", "nasdaq-hc-tech"])
    assert args.universe == "nasdaq-hc-tech" and args.tickers is None


def test_backtest_universe_mode_resolves_from_store(tmp_path, monkeypatch):
    """main() backtest mode + --universe resolves config.tickers from the cached
    SIC store (offline: fetch_listed_symbols and run are monkeypatched)."""
    from types import SimpleNamespace

    from store.schema import COLUMNS
    from store.store import PITStore

    import run as run_mod

    db = tmp_path / "bt.sqlite"
    PITStore(db).put_data(pd.DataFrame(
        [_sic_row("TECHX", 7372), _sic_row("HEALX", 2836), _sic_row("BANKX", 6022)],
        columns=COLUMNS))

    monkeypatch.setattr(run_mod, "fetch_listed_symbols",
                        lambda: ["TECHX", "HEALX", "BANKX"])
    captured = {}

    def _fake_run(config):
        captured["tickers"] = config.tickers
        return SimpleNamespace(
            report=SimpleNamespace(verdict="KILL"), report_path="x", statuses={})

    monkeypatch.setattr(run_mod, "run", _fake_run)
    rc = run_mod.main(["--db", str(db), "--universe", "nasdaq-hc-tech",
                       "--start-year", "2018", "--end-year", "2020", "--signals", "momentum"])
    assert rc == 0
    assert set(captured["tickers"]) == {"TECHX", "HEALX"}  # point-in-time hc+tech only


def test_report_states_survivor_limited(tmp_path):
    store, winners, losers = _build_store(tmp_path)
    res = run(_config(tmp_path, store, winners, losers))
    text = Path(res.report_path).read_text()
    assert "SURVIVOR-LIMITED" in text and "$0 data gap" in text  # §2.2 said plainly


def test_min_obs_per_slice_override_allows_small_slice(tmp_path):
    store, winners, losers = _build_store(tmp_path, n_win=3, n_lose=2)  # 5 obs per year
    # the strict default floor (~20) skips every small slice -> nothing evaluable
    with pytest.raises(PipelineError):
        run(_config(tmp_path, store, winners, losers, min_samples_per_arm=3,
                    two_arm_kwargs={"threshold": 60.0, "n_boot": 100}))
    # lowering the floor lets the small slice evaluate and produce a verdict
    res = run(_config(tmp_path, store, winners, losers, min_samples_per_arm=3,
                      two_arm_kwargs={"threshold": 60.0, "n_boot": 100, "min_obs_per_slice": 4}))
    assert res.report.verdict in {"GO", "MARGINAL", "KILL"}
