"""Rank-as-of-date funnel (ARCHITECTURE.md §2 no-peek, §6).

For an as-of date, score every universe ticker on the composite of the requested
signals — strictly point-in-time (the scorers read only ``store.get_data(...,
as_of)``) — rank descending, and emit a markdown table of the top N with the
composite, the per-signal sub-scores, a percentile rank, and (for past dates) the
realized 12-month forward return as CONTEXT.

This is a RANKING, not the §3 two-arm experiment: a percentile rank is explicitly
NOT a win probability, and every table header says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import mean
from typing import Callable

from backtest.labels import BENCHMARK_TICKER, label_winner


@dataclass
class RankRow:
    rank: int
    ticker: str
    composite: float
    subscores: dict          # signal name -> sub-score (or None)
    percentile: float        # 100 = highest composite in the as-of universe
    fwd_return: float | None  # realized 12m return, or None (n/a — missing data)


@dataclass
class RankingResult:
    as_of: date
    rows: list[RankRow]      # top N
    n_universe: int          # names considered
    n_scored: int            # names with at least one sub-score


def _forward_return(ticker, as_of, store, benchmark):
    """Realized 12-month forward return of the ticker, or None when the window
    has not elapsed / data is missing. Reuses the labeler (no peeking into the
    ranking — this is context only)."""
    label = label_winner(ticker, as_of, store=store, benchmark=benchmark)
    return label.components.get("ticker_return") if label.outcome_known else None


def rank_as_of(
    as_of: date,
    tickers,
    *,
    store,
    scorers: dict[str, Callable],
    top_n: int = 25,
    benchmark: str = BENCHMARK_TICKER,
    with_forward_return: bool = True,
) -> RankingResult:
    """Score, rank, and percentile the universe as-of ``as_of``. ``scorers`` maps
    signal name -> fn(ticker, as_of, store) -> score | None."""
    scored = []
    for t in tickers:
        subs = {name: fn(t, as_of, store) for name, fn in scorers.items()}
        present = [v for v in subs.values() if v is not None]
        if not present:
            continue  # no scorable signal -> not rankable
        scored.append((t, mean(present), subs))

    scored.sort(key=lambda x: x[1], reverse=True)  # highest composite first
    n = len(scored)
    rows = []
    for i, (t, comp, subs) in enumerate(scored[:top_n]):
        percentile = 100.0 * (n - i) / n if n else float("nan")  # 100 = top
        fwd = _forward_return(t, as_of, store, benchmark) if with_forward_return else None
        rows.append(RankRow(i + 1, t, comp, subs, percentile, fwd))
    return RankingResult(as_of=as_of, rows=rows, n_universe=len(tickers), n_scored=n)


def render_ranking_markdown(results: list[RankingResult], *, coverage: dict,
                            signal_names) -> str:
    """Render the per-as-of ranking tables + a coverage section. ``coverage`` has
    n_universe / n_priced / n_classified / n_quarantined."""
    sig_cols = list(signal_names)
    lines = ["# Rank-as-of-date funnel — Nasdaq healthcare + technology", ""]

    lines += [
        "> SURVIVORSHIP NOTE: the universe is built from today's listed symbols "
        "(free feeds lack delisted names — the $0 gap), so historical membership "
        "is SURVIVOR-LIMITED, not survivorship-free (ARCHITECTURE.md §2.2).",
        "",
        "## Coverage",
        f"- universe names: {coverage.get('n_universe', 'n/a')}",
        f"- classified (tech/healthcare): {coverage.get('n_classified', 'n/a')}",
        f"- priced: {coverage.get('n_priced', 'n/a')}",
        f"- quarantined: {coverage.get('n_quarantined', 'n/a')}",
        "",
    ]

    header = "| rank | ticker | composite | " + " | ".join(sig_cols) + " | percentile | fwd 12m ret |"
    divider = "|" + "---|" * (4 + len(sig_cols) + 1)
    for r in results:
        lines.append(f"## Ranking as-of {r.as_of} — a percentile rank, not a win-probability")
        lines.append(f"(universe {r.n_universe}, scored {r.n_scored}, showing top {len(r.rows)})")
        lines.append(header)
        lines.append(divider)
        for row in r.rows:
            subs = " | ".join(
                ("n/a" if row.subscores.get(s) is None else f"{row.subscores[s]:.1f}")
                for s in sig_cols
            )
            fwd = "n/a" if row.fwd_return is None else f"{row.fwd_return:+.1%}"
            lines.append(
                f"| {row.rank} | {row.ticker} | {row.composite:.1f} | {subs} "
                f"| {row.percentile:.1f} | {fwd} |"
            )
        lines.append("")
    return "\n".join(lines)
