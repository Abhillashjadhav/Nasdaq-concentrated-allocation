"""stockscope orchestrator — plain Python, never an LLM (ARCHITECTURE.md §5, §7).

``run(config)`` sequences the whole harness end to end:

    ingest (optional, available adapters) -> resolve universe per entry date ->
    score active signals per (ticker x date) -> label outcomes (the sanctioned
    look-ahead, cached) -> assemble a Pandera-validated observation frame per
    signal -> walk-forward (two-arm per slice) -> build_report -> write report.md

Fail loud, never fake (§2, CLAUDE.md)
-------------------------------------
* A signal whose data ADAPTER is not wired is marked ``not_run`` — distinct from
  ``insufficient_data`` (adapter exists, but the store lacks enough data).
* If NO signal is runnable, or none yields an evaluable result (e.g. the
  benchmark series is absent so every label is not-yet-known), ``run`` raises
  ``PipelineError`` rather than emit a verdict on data it does not have.

Config-driven scope (``RunConfig``) lets you run a tiny slice (a few tickers, two
entry years, quality only) and scale up with no code change. Signal scorers are
injectable; the default registry wires the real signals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from backtest.labels import BENCHMARK_TICKER, label_winner
from backtest.walk_forward import run_walk_forward
from report.build_report import build_report
from signals.insider import insider_cluster_score
from signals.momentum import momentum_score
from signals.quality import quality_score
from signals.revisions import revision_breadth_score
from stats.two_arm import OBSERVATION_SCHEMA
from universe.universe import build_universe

OBS_COLS = ["ticker", "as_of", "score", "is_winner", "excess_return"]


class PipelineError(RuntimeError):
    """Raised when the run cannot honestly produce a verdict."""


@dataclass
class SignalSpec:
    name: str
    # score(ticker, as_of, store) -> (score | None, insufficient_data: bool)
    score: Callable[[str, date, object], tuple[float | None, bool]]
    adapter_available: bool  # is there a wired ingest path for this signal's data?


def _wrap(fn):
    """Adapt a signal's score function to the (score, insufficient) contract."""
    def scorer(ticker, as_of, store):
        r = fn(ticker, as_of, store=store)
        return r.score, r.insufficient_data
    return scorer


# Default registry. momentum (prices adapter), quality (EDGAR adapter) and
# insiders (Form 4 adapter, data/form4.py) have a live ingest path; revisions
# (estimate snapshots) does not yet, so it is marked not_run until its adapter lands.
DEFAULT_SIGNALS: dict[str, SignalSpec] = {
    "momentum": SignalSpec("momentum", _wrap(momentum_score), adapter_available=True),
    "quality": SignalSpec("quality", _wrap(quality_score), adapter_available=True),
    "revisions": SignalSpec("revisions", _wrap(revision_breadth_score), adapter_available=False),
    "insiders": SignalSpec("insiders", _wrap(insider_cluster_score), adapter_available=True),
}


@dataclass
class RunConfig:
    store: object
    tickers: list[str]
    entry_dates: list[date]
    active_signals: list[str]
    output_dir: str
    benchmark: str = BENCHMARK_TICKER
    apply_liquidity_filter: bool = False
    sector_classifier: object = None
    signals: dict[str, SignalSpec] | None = None  # defaults to DEFAULT_SIGNALS
    ingest: bool = False  # if True, call live adapters to populate the store first
    survivorship_haircut_pp: float = 4.0
    min_consistency_years: int = 3
    min_samples_per_arm: int = 300
    two_arm_kwargs: dict = field(default_factory=dict)


@dataclass
class RunResult:
    report: object
    statuses: dict[str, str]            # signal -> "evaluated" | "not_run" | "insufficient_data: ..."
    observations: dict[str, pd.DataFrame]
    report_path: str
    not_run: list[str]
    ingest_quarantine: list[dict] = field(default_factory=list)  # tickers skipped at ingest


def _resolve_universe(as_of, config, store) -> list[str]:
    if config.apply_liquidity_filter:
        return build_universe(
            as_of, candidates=config.tickers, store=store,
            sector_classifier=config.sector_classifier,
        ).tickers
    return list(config.tickers)


def _ingest(config, store) -> list[dict]:
    """Best-effort live ingest via the available adapters (network). Only runs when
    config.ingest is True. A per-ticker price failure (e.g. a delisted name with no
    free data) is caught, recorded as a quarantined coverage gap, and skipped so the
    rest of the universe proceeds; the run fails only if NO universe ticker ingests.
    The same per-ticker resilience applies to fundamentals and Form 4 (a delisted /
    transient failure quarantines that ticker's data for that source, never aborts).
    Config errors (e.g. a missing SEC User-Agent) still propagate — that is setup,
    not a per-ticker data gap. Returns the quarantine records (evals.coverage shape)."""
    # Local imports keep network deps out of import time AND make the adapters
    # monkeypatchable per-call in tests (rebound from their modules each call).
    from data.edgar_client import (  # per-ticker EDGAR errors + shared client/resolver
        CikResolver, EdgarClient, EdgarHTTPError, UnknownTickerError,
    )
    from data.form4 import fetch_insider_buys
    from data.fundamentals import fetch_fundamentals  # local import: network deps
    from data.prices import DataPullError, fetch_prices

    edgar_errs = (UnknownTickerError, EdgarHTTPError)

    start = min(config.entry_dates).replace(year=min(d.year for d in config.entry_dates) - 2)
    end = max(config.entry_dates).replace(year=max(d.year for d in config.entry_dates) + 1)

    quarantine: list[dict] = []
    n_priced = 0
    for ticker in config.tickers:
        try:
            store.put_data(fetch_prices(ticker, start, end))
            n_priced += 1
        except DataPullError as exc:  # delisted / no free data -> quarantine, don't abort
            quarantine.append({"ticker": ticker, "field": "close",
                               "reason": f"price_unavailable: {exc}", "vendor": "prices"})
    if n_priced == 0:
        raise PipelineError(
            "ingest produced no priced tickers (every fetch_prices failed); "
            "refusing to proceed"
        )

    # The benchmark is needed to grade; quarantine (not abort) if it fails — the
    # labeler then surfaces the gap downstream.
    try:
        store.put_data(fetch_prices(config.benchmark, start, end))
    except DataPullError as exc:
        quarantine.append({"ticker": config.benchmark, "field": "close",
                           "reason": f"benchmark_price_unavailable: {exc}", "vendor": "prices"})

    # One EDGAR client + resolver for ALL tickers, so a single throttle governs the
    # AGGREGATE request rate across the whole run (SEC rate-limits by IP, not per
    # ticker). The resolver's company_tickers map is also fetched just once.
    edgar = EdgarClient()
    resolver = CikResolver(edgar)
    for ticker in config.tickers:
        try:
            fetch_fundamentals(ticker, client=edgar, resolver=resolver, store=store, write=True)
        except edgar_errs as exc:  # delisted / transient -> quarantine, don't abort
            quarantine.append({"ticker": ticker, "field": "fundamentals",
                               "reason": f"fundamentals_unavailable: {exc}", "vendor": "edgar"})

    # Form 4 needs only filings shortly before each entry (the insider lookback is
    # ~90 days), NOT the wide price window (which reaches back min_year-2 for the
    # momentum history). Bound it tightly so a 2019-2021 run doesn't pull 2017
    # filings: from ~6 months before the first entry through the last entry.
    f4_start = pd.Timestamp(min(config.entry_dates)) - pd.Timedelta(days=180)
    f4_end = pd.Timestamp(max(config.entry_dates))
    for ticker in config.tickers:
        try:
            res = fetch_insider_buys(ticker, client=edgar, resolver=resolver,
                                     store=store, write=True, start=f4_start, end=f4_end)
            quarantine.extend(getattr(res, "gaps", []))  # per-filing quarantines surfaced
        except edgar_errs as exc:
            quarantine.append({"ticker": ticker, "field": "form4_buy_P",
                               "reason": f"form4_unavailable: {exc}", "vendor": "edgar"})
    return quarantine


def run(config: RunConfig) -> RunResult:
    """Run the end-to-end pipeline and write the GO/KILL report. See module docstring."""
    store = config.store
    signals = config.signals or DEFAULT_SIGNALS
    ingest_quarantine = _ingest(config, store) if config.ingest else []

    statuses: dict[str, str] = {}
    runnable: list[tuple[str, SignalSpec]] = []
    for name in config.active_signals:
        spec = signals.get(name)
        if spec is None:
            statuses[name] = "unknown_signal"
        elif not spec.adapter_available:
            statuses[name] = "not_run"  # no live adapter — do not pretend
        else:
            statuses[name] = "evaluated"  # provisional
            runnable.append((name, spec))

    if not runnable:
        raise PipelineError(
            "no runnable signals: every active signal lacks a live adapter; "
            "refusing to emit a verdict on data we do not have"
        )

    label_cache: dict[tuple, object] = {}

    def _label(ticker, as_of):
        key = (ticker, as_of)
        if key not in label_cache:
            label_cache[key] = label_winner(
                ticker, as_of, store=store, benchmark=config.benchmark
            )
        return label_cache[key]

    observations: dict[str, pd.DataFrame] = {}
    for name, spec in runnable:
        rows = []
        for as_of in config.entry_dates:
            for ticker in _resolve_universe(as_of, config, store):
                score, insufficient = spec.score(ticker, as_of, store)
                lbl = _label(ticker, as_of)
                rows.append({
                    "ticker": ticker,
                    "as_of": pd.Timestamp(as_of),
                    "score": None if insufficient else float(score),
                    "is_winner": lbl.is_winner,  # True / False / None (not-yet-known)
                    "excess_return": (lbl.components.get("excess_return")
                                      if lbl.outcome_known else None),
                })
        # Pandera-validate the assembled frame at the module boundary (§8).
        observations[name] = OBSERVATION_SCHEMA.validate(
            pd.DataFrame(rows, columns=OBS_COLS), lazy=True
        )

    wf_results = {}
    for name, df in observations.items():
        try:
            wf_results[name] = run_walk_forward(df, **config.two_arm_kwargs)
        except ValueError as exc:
            statuses[name] = f"insufficient_data: {exc}"

    if not wf_results:
        raise PipelineError(
            "no signal produced an evaluable result (insufficient labeled data "
            "after purge/embargo); refusing to emit a verdict"
        )

    report = build_report(
        wf_results,
        survivorship_haircut_pp=config.survivorship_haircut_pp,
        min_consistency_years=config.min_consistency_years,
        min_samples_per_arm=config.min_samples_per_arm,
    )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text(report.markdown)

    return RunResult(
        report=report, statuses=statuses, observations=observations,
        report_path=str(path), not_run=[n for n, s in statuses.items() if s == "not_run"],
        ingest_quarantine=ingest_quarantine,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockscope",
        description="Winner-signal backtest harness (GO/KILL). See ARCHITECTURE.md.",
    )
    parser.add_argument("--db", required=True, help="path to the point-in-time store SQLite file")
    parser.add_argument("--tickers", required=True, help="comma-separated candidate tickers")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--signals", default="momentum,quality", help="comma-separated active signals")
    parser.add_argument("--output", default="outputs", help="directory for report.md")
    parser.add_argument("--liquidity-filter", action="store_true")
    parser.add_argument("--ingest", action="store_true", help="call live adapters to populate the store")
    parser.add_argument(
        "--min-obs-per-slice", type=int, default=None,
        help="override the walk-forward per-slice observation floor (default ~20, "
             "strict — lower it only for small validation slices)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from store.store import PITStore

    # Only override the strict default floor when the flag is given.
    two_arm_kwargs = ({} if args.min_obs_per_slice is None
                      else {"min_obs_per_slice": args.min_obs_per_slice})
    config = RunConfig(
        store=PITStore(args.db),
        tickers=[t.strip() for t in args.tickers.split(",") if t.strip()],
        entry_dates=[date(y, 1, 1) for y in range(args.start_year, args.end_year + 1)],
        active_signals=[s.strip() for s in args.signals.split(",") if s.strip()],
        output_dir=args.output,
        apply_liquidity_filter=args.liquidity_filter,
        ingest=args.ingest,
        two_arm_kwargs=two_arm_kwargs,
    )
    result = run(config)
    print(f"VERDICT: {result.report.verdict}  ->  {result.report_path}")
    print(f"signals: {result.statuses}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
