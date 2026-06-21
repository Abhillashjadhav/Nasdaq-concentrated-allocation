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
from report.ranking import rank_as_of, render_ranking_markdown
from signals.momentum import momentum_score
from signals.quality import quality_score
from signals.revisions import revision_breadth_score
from stats.two_arm import OBSERVATION_SCHEMA
from universe.hc_tech import classify_and_cache, nasdaq_hc_tech_universe
from universe.nasdaq_directory import fetch_listed_symbols
from universe.universe import build_universe

OBS_COLS = ["ticker", "as_of", "score", "is_winner", "excess_return"]
_FAR_FUTURE = date(2100, 1, 1)
_SIGNAL_FNS = {
    "momentum": momentum_score, "quality": quality_score,
    "revisions": revision_breadth_score,
}


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


# Default registry. momentum (prices adapter) and quality (SimFin bulk adapter)
# have a live ingest path; revisions (estimate snapshots) does not yet, so it is
# marked not_run until its adapter lands.
DEFAULT_SIGNALS: dict[str, SignalSpec] = {
    "momentum": SignalSpec("momentum", _wrap(momentum_score), adapter_available=True),
    "quality": SignalSpec("quality", _wrap(quality_score), adapter_available=True),
    "revisions": SignalSpec("revisions", _wrap(revision_breadth_score), adapter_available=False),
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
    refresh_fundamentals: bool = False  # force re-download of the SimFin bulk cache
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
    Fundamentals come from SimFin's bulk download (one pull covers the whole
    universe); names SimFin does not cover are quarantined, never fatal. Config
    errors (e.g. a missing SimFin API key) still propagate — that is setup, not a
    per-ticker data gap. Returns the quarantine records (evals.coverage shape)."""
    # Local imports keep network deps out of import time AND make the adapters
    # monkeypatchable per-call in tests (rebound from their modules each call).
    from data.prices import DataPullError, fetch_prices
    from data.simfin_client import load_simfin_fundamentals

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

    # Fundamentals: a single SimFin bulk download covers the whole universe,
    # sidestepping the per-ticker throttle that left quality n/a at scale. The ticker
    # filter is pushed into the loader so only the universe's rows are processed.
    res = load_simfin_fundamentals(
        store, refresh=config.refresh_fundamentals, tickers=config.tickers,
    )
    quarantine.extend(res.quarantine)  # names SimFin doesn't cover, surfaced
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


def _make_scorer(fn):
    def scorer(ticker, as_of, store):  # -> composite sub-score | None
        r = fn(ticker, as_of, store=store)
        return None if r.insufficient_data else r.score
    return scorer


def _signal_scorers(active_signals) -> dict:
    return {n: _make_scorer(_SIGNAL_FNS[n]) for n in active_signals if n in _SIGNAL_FNS}


def resolve_universe_candidates(symbols, store, as_of) -> list[str]:
    """Backtest/GO-KILL candidates from the real (survivor-limited) hc+tech
    universe: the classified members reporting as-of ``as_of`` (read point-in-time
    from the cached sector store). Fail loud if the cache is empty — the universe
    must be built first (``--universe nasdaq-hc-tech --rank-asof ... --ingest``),
    never silently run the verdict on zero names."""
    members = nasdaq_hc_tech_universe(as_of, symbols, store=store)
    if not members:
        raise PipelineError(
            "no classified hc+tech names in the store as-of "
            f"{as_of} — build the universe first (--universe nasdaq-hc-tech "
            "--rank-asof ... --ingest) or pass an explicit --tickers list; "
            "refusing to emit a verdict on an empty universe"
        )
    return members


@dataclass
class RankingRun:
    results: list
    coverage: dict
    report_path: str


def run_ranking(config, *, asof_dates, top_n, symbols, n_quarantined=0, scorers=None,
                all_tickers=False):
    """Rank the tech+healthcare universe as-of each date and write ranking.md.
    ``scorers`` defaults to the active signals' point-in-time scorers.

    When ``all_tickers`` is True the sector/universe filter is skipped: candidates
    are every ticker that has price data (``field='close'``) in the store, so names
    with fundamentals but no cached ``sector`` row are scored too. No-peek is intact
    — each candidate is still scored through ``store.get_data(..., as_of)``."""
    store = config.store
    scorers = scorers if scorers is not None else _signal_scorers(config.active_signals)
    if not scorers:
        raise PipelineError("ranking needs at least one scorable signal in --signals")

    if all_tickers:
        priced = store.tickers_with_field("close")
        if not priced:
            raise PipelineError(
                "--all-tickers: no tickers with price data (field='close') in the "
                "store; ingest prices first or drop --all-tickers"
            )
        members_for = lambda ao: priced  # noqa: E731 — same candidate set per date
        n_classified = len(priced)
        n_priced = len(priced)
        n_universe = len(priced)
    else:
        members_for = lambda ao: nasdaq_hc_tech_universe(ao, symbols, store=store)  # noqa: E731
        classified = nasdaq_hc_tech_universe(_FAR_FUTURE, symbols, store=store)
        n_classified = len(classified)
        n_priced = sum(1 for s in classified if not store.get_data("close", s, _FAR_FUTURE).empty)
        n_universe = len(symbols)

    results = [
        rank_as_of(ao, members_for(ao),
                   store=store, scorers=scorers, top_n=top_n, benchmark=config.benchmark)
        for ao in asof_dates
    ]
    coverage = {"n_universe": n_universe, "n_classified": n_classified,
                "n_priced": n_priced, "n_quarantined": n_quarantined}

    md = render_ranking_markdown(results, coverage=coverage,
                                 signal_names=[n for n in config.active_signals if n in scorers])
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ranking.md"
    path.write_text(md)
    return RankingRun(results=results, coverage=coverage, report_path=str(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockscope",
        description="Winner-signal backtest harness (GO/KILL). See ARCHITECTURE.md.",
    )
    parser.add_argument("--db", required=True, help="path to the point-in-time store SQLite file")
    parser.add_argument("--tickers", default=None,
                        help="comma-separated candidate tickers (backtest mode)")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--signals", default="momentum,quality", help="comma-separated active signals")
    parser.add_argument("--output", default="outputs", help="directory for the report")
    parser.add_argument("--liquidity-filter", action="store_true")
    parser.add_argument("--ingest", action="store_true", help="call live adapters to populate the store")
    parser.add_argument("--refresh-fundamentals", action="store_true",
                        help="force re-download of SimFin bulk data (ignore the local cache)")
    parser.add_argument(
        "--min-obs-per-slice", type=int, default=None,
        help="override the walk-forward per-slice observation floor (default ~20, "
             "strict — lower it only for small validation slices)",
    )
    # rank-as-of-date funnel
    parser.add_argument("--universe", choices=["nasdaq-hc-tech"], default=None,
                        help="use the real Nasdaq healthcare+tech universe: with "
                             "--rank-asof it ranks; in backtest mode (no --rank-asof) "
                             "it runs the GO/KILL verdict on that universe")
    parser.add_argument("--rank-asof", action="append", type=date.fromisoformat, default=None,
                        metavar="YYYY-MM-DD", help="rank the universe as-of this date (repeatable)")
    parser.add_argument("--top-n", type=int, default=25, help="rows to show per ranking table")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="force re-classification of the universe (ignore the cached sector)")
    parser.add_argument("--universe-limit", type=int, default=None,
                        help="cap the universe size for a fast smoke test before a full run")
    parser.add_argument("--all-tickers", action="store_true",
                        help="ranking mode: skip the sector/universe filter and score "
                             "EVERY ticker with price data in the store (not just the "
                             "classified hc+tech names)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from store.store import PITStore

    signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    # --- ranking mode: rank the real universe as-of date(s) ------------------
    if args.rank_asof:
        if not args.all_tickers and args.universe != "nasdaq-hc-tech":
            parser.error("--rank-asof requires --universe nasdaq-hc-tech (or --all-tickers)")
        store = PITStore(args.db)
        config = RunConfig(store=store, tickers=[], entry_dates=list(args.rank_asof),
                           active_signals=signals, output_dir=args.output, ingest=args.ingest,
                           refresh_fundamentals=args.refresh_fundamentals)
        if args.all_tickers:
            # Score everything with price data already in the store — no universe
            # fetch, no sector classification, no ingest gate.
            rr = run_ranking(config, asof_dates=list(args.rank_asof), top_n=args.top_n,
                             symbols=[], n_quarantined=0, all_tickers=True)
            print(f"RANKING -> {rr.report_path}  (coverage: {rr.coverage})")
            return 0
        symbols = fetch_listed_symbols()
        if args.universe_limit is not None:
            symbols = symbols[:args.universe_limit]  # cap for a fast smoke test
        n_quarantined = 0
        if args.ingest:
            cres = classify_and_cache(symbols, store=store, refresh=args.refresh_universe)
            config.tickers = nasdaq_hc_tech_universe(max(args.rank_asof), symbols, store=store)
            n_quarantined = cres.n_quarantined + len(_ingest(config, store))
        rr = run_ranking(config, asof_dates=list(args.rank_asof), top_n=args.top_n,
                         symbols=symbols, n_quarantined=n_quarantined)
        print(f"RANKING -> {rr.report_path}  (coverage: {rr.coverage})")
        return 0

    # --- backtest mode (GO/KILL) ---------------------------------------------
    if not args.tickers and args.universe != "nasdaq-hc-tech":
        parser.error("backtest mode needs --tickers or --universe nasdaq-hc-tech")

    store = PITStore(args.db)
    entry_dates = [date(y, 1, 1) for y in range(args.start_year, args.end_year + 1)]
    if args.universe == "nasdaq-hc-tech":
        # Run the verdict on the real (survivor-limited) universe instead of a
        # hand-typed list: resolve membership point-in-time as-of the last entry.
        symbols = fetch_listed_symbols()
        if args.universe_limit is not None:
            symbols = symbols[:args.universe_limit]  # cap for a fast smoke test
        if args.ingest:
            # Populate the sector classification cache before resolving from it —
            # an empty store has no classified names yet.
            classify_and_cache(symbols, store=store, refresh=args.refresh_universe)
        tickers = resolve_universe_candidates(symbols, store, max(entry_dates))
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    # Only override the strict default floor when the flag is given.
    two_arm_kwargs = ({} if args.min_obs_per_slice is None
                      else {"min_obs_per_slice": args.min_obs_per_slice})
    config = RunConfig(
        store=store,
        tickers=tickers,
        entry_dates=entry_dates,
        active_signals=signals,
        output_dir=args.output,
        apply_liquidity_filter=args.liquidity_filter,
        ingest=args.ingest,
        two_arm_kwargs=two_arm_kwargs,
        refresh_fundamentals=args.refresh_fundamentals,
    )
    result = run(config)
    print(f"VERDICT: {result.report.verdict}  ->  {result.report_path}")
    print(f"signals: {result.statuses}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
