"""Report + GO/KILL verdict (ARCHITECTURE.md §3, §10, §9.13).

``build_report`` assembles the walk-forward output (per-signal, per-year lift / CI
/ rank-IC + aggregate) and the coverage report, renders a human-readable Markdown
summary, and ends with an explicit GO / MARGINAL / KILL verdict computed exactly
against the §10 thresholds.

Verdict logic (faithful to §10 — the threshold is the threshold)
----------------------------------------------------------------
A signal/archetype CONFIRMS only if ALL of:
  1. it holds in >= ``min_consistency_years`` distinct years — i.e. that many
     year-slices whose lift CI excludes zero on the positive side (the honest,
     post-purge significance), and
  2. its mean lift survives the survivorship/cost haircut: ``mean_lift`` minus
     ``survivorship_haircut_pp`` is still > 0 (default 4pp — the conservative end
     of the 1–4pp band, §3), and
  3. rank-IC is positive (higher score -> higher forward excess return), and
  4. the §8 sample-size floor is met: >= ``min_samples_per_arm`` observations in
     BOTH arms (fired and not-fired), pooled across the confirming slices.

Overall verdict (§10):
  * GO       — >= 2 archetypes confirm.
  * MARGINAL — exactly 1 confirms (the §10 "Partial": narrow the build to it).
  * KILL     — 0 confirm (lift not significant in >=3 years -> abandon the thesis).

A marginal or thin result is NEVER rounded up to GO. Sub-haircut lift, too few
years, a non-positive IC, or an under-floor sample each block confirmation.

Honesty caveats are rendered IN the report (coverage gaps, effective de-overlapped
sample size, the haircut, and the out-of-scope qualitative/LLM layer) so the
verdict cannot be read as more certain than the data supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from backtest.walk_forward import WalkForwardResult

GO_MIN_ARCHETYPES = 2          # §10: holds across >= 2 archetypes
DEFAULT_HAIRCUT_PP = 4.0       # §3: survives a 1–4pp survivorship haircut (conservative end)
DEFAULT_CONSISTENCY_YEARS = 3  # §3/§10: holds in >= 3 distinct years
DEFAULT_MIN_SAMPLES_PER_ARM = 300  # §8: ~300/arm sample-size floor


@dataclass
class SignalVerdict:
    signal: str
    confirmed: bool
    n_sig_years: int
    mean_lift: float
    haircut_adjusted_lift: float
    mean_rank_ic: float
    n_fired: int
    n_not_fired: int
    sample_floor_met: bool
    reasons: list[str]  # why it failed (empty if confirmed)


@dataclass
class Report:
    verdict: str  # "GO" | "MARGINAL" | "KILL"
    n_confirmed: int
    signal_verdicts: list[SignalVerdict]
    survivorship_haircut_pp: float
    caveats: list[str] = dc_field(default_factory=list)
    markdown: str = ""


def evaluate_signal(
    signal: str,
    wf: WalkForwardResult,
    *,
    survivorship_haircut_pp: float = DEFAULT_HAIRCUT_PP,
    min_consistency_years: int = DEFAULT_CONSISTENCY_YEARS,
    min_samples_per_arm: int = DEFAULT_MIN_SAMPLES_PER_ARM,
) -> SignalVerdict:
    """Apply the §10 GO criteria to one signal's walk-forward result."""
    n_sig = wf.n_years_significant_positive
    mean_lift = wf.mean_lift
    adj = mean_lift - survivorship_haircut_pp / 100.0
    ic = wf.mean_rank_ic
    n_fired = sum(s.n_fired for s in wf.slices)
    n_not_fired = sum(s.n_not_fired for s in wf.slices)
    floor_met = min(n_fired, n_not_fired) >= min_samples_per_arm

    reasons = []
    if n_sig < min_consistency_years:
        reasons.append(f"holds in only {n_sig} yr (< {min_consistency_years})")
    if not (adj > 0):
        reasons.append(f"lift {mean_lift:.3f} does not survive {survivorship_haircut_pp:.0f}pp haircut")
    if not (ic > 0):
        reasons.append(f"rank-IC not positive ({ic:.3f})")
    if not floor_met:
        reasons.append(f"sample floor not met (min arm {min(n_fired, n_not_fired)} < {min_samples_per_arm})")

    return SignalVerdict(
        signal=signal, confirmed=not reasons, n_sig_years=n_sig, mean_lift=mean_lift,
        haircut_adjusted_lift=adj, mean_rank_ic=ic, n_fired=n_fired,
        n_not_fired=n_not_fired, sample_floor_met=floor_met, reasons=reasons,
    )


def resolve_verdict(n_confirmed: int) -> str:
    """§10: >=2 confirm -> GO; exactly 1 -> MARGINAL (Partial); 0 -> KILL."""
    if n_confirmed >= GO_MIN_ARCHETYPES:
        return "GO"
    if n_confirmed == 1:
        return "MARGINAL"
    return "KILL"


def build_report(
    results: dict[str, WalkForwardResult],
    *,
    coverage_by_year: dict | None = None,
    survivorship_haircut_pp: float = DEFAULT_HAIRCUT_PP,
    min_consistency_years: int = DEFAULT_CONSISTENCY_YEARS,
    min_samples_per_arm: int = DEFAULT_MIN_SAMPLES_PER_ARM,
) -> Report:
    """Assemble per-signal verdicts into the GO/MARGINAL/KILL report. ``results``
    maps signal/archetype name -> its WalkForwardResult."""
    if not results:
        raise ValueError("no signals to report on")

    verdicts = [
        evaluate_signal(
            name, wf, survivorship_haircut_pp=survivorship_haircut_pp,
            min_consistency_years=min_consistency_years,
            min_samples_per_arm=min_samples_per_arm,
        )
        for name, wf in results.items()
    ]
    n_confirmed = sum(v.confirmed for v in verdicts)
    verdict = resolve_verdict(n_confirmed)
    caveats = _caveats(results, survivorship_haircut_pp)
    markdown = _render(verdict, n_confirmed, verdicts, results, coverage_by_year,
                       survivorship_haircut_pp, caveats)

    return Report(
        verdict=verdict, n_confirmed=n_confirmed, signal_verdicts=verdicts,
        survivorship_haircut_pp=survivorship_haircut_pp, caveats=caveats, markdown=markdown,
    )


def _caveats(results, haircut_pp) -> list[str]:
    raw = sum(wf.n_observations_raw for wf in results.values())
    kept = sum(wf.n_observations_kept for wf in results.values())
    years = sorted({s.year for wf in results.values() for s in wf.slices})
    return [
        "SURVIVOR-LIMITED universe (§2.2): the free Nasdaq-Trader feed lists only "
        "today's survivors, so delisted/merged/bankrupt names are ABSENT (the $0 "
        "data gap). This is NOT survivorship-free; survivorship inflates win-rates "
        "~1–4pp/yr, which the haircut below is meant to offset, not erase.",
        f"Survivorship/cost haircut of {haircut_pp:.0f}pp applied to every lift; "
        f"the reported edge is net of this conservative drag.",
        f"Significance is post-purge/embargo on de-overlapped yearly slices: "
        f"effective sample = {len(years)} independent year-slices, not the "
        f"{kept} kept (of {raw} raw) pooled observations — pooling overlapping "
        f"12-month labels would inflate significance.",
        "Coverage gaps (especially delisted tickers and fundamentals) shrink the "
        "universe each year — see the coverage section; a thin year is less certain.",
        "The Bull/Critic/Reconciler LLM debate and other qualitative layers are "
        "OUT OF SCOPE (§11); this verdict rests only on the quantitative two-arm "
        "evidence and must not be read as more certain than that.",
    ]


def _render(verdict, n_confirmed, verdicts, results, coverage_by_year, haircut_pp, caveats) -> str:
    lines = [
        "# Winner-Signal Backtest — GO/KILL Report",
        "",
        f"## VERDICT: {verdict}",
        f"{n_confirmed} of {len(verdicts)} archetype(s) confirm "
        f"(GO needs >= {GO_MIN_ARCHETYPES}; MARGINAL = 1; KILL = 0).",
        "",
        "## Per-signal summary",
        "| signal | confirms | sig-yrs | mean lift | net of haircut | rank-IC | fired/not | floor |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v.signal} | {'YES' if v.confirmed else 'no'} | {v.n_sig_years} "
            f"| {v.mean_lift:+.3f} | {v.haircut_adjusted_lift:+.3f} | {v.mean_rank_ic:+.3f} "
            f"| {v.n_fired}/{v.n_not_fired} | {'ok' if v.sample_floor_met else 'LOW'} |"
        )
        if v.reasons:
            lines.append(f"|   ↳ blocked: {'; '.join(v.reasons)} |||||||| ")

    lines += ["", "## Per-year detail"]
    for name, wf in results.items():
        lines.append(f"### {name}")
        lines.append("| year | n | lift | CI | rank-IC | sig+ |")
        lines.append("|---|---|---|---|---|---|")
        for s in wf.slices:
            lines.append(
                f"| {s.year} | {s.n} | {s.lift:+.3f} | "
                f"[{s.ci_low:+.3f}, {s.ci_high:+.3f}] | {s.rank_ic:+.3f} | "
                f"{'Y' if s.significant_positive else '-'} |"
            )

    lines += ["", "## Coverage (the $0 asterisk made visible)"]
    if coverage_by_year:
        for year, cov in sorted(coverage_by_year.items()):
            missing = getattr(cov, "missing", [])
            lines.append(f"- {year}: {len(missing)} coverage gap(s)")
    else:
        lines.append("- (no coverage report supplied)")

    lines += ["", "## Honesty caveats"] + [f"- {c}" for c in caveats]
    return "\n".join(lines)
