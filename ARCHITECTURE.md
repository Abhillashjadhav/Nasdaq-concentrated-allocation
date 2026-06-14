# ARCHITECTURE.md — Winner-Signal Backtest Harness

Working repo name: `stockscope`. This document is the source of truth. The
pr-reviewer agent judges every PR against it.

## 1. Purpose

This repo answers one question before any capital or any live system is built:

> Do a small set of proven signals, measured point-in-time on a
> survivorship-free universe, separate future winners from losers in US Nasdaq
> healthcare + technology stocks — with a statistically significant edge over
> the base rate, holding across ≥3 distinct years?

* **GO** (signals work) → this harness becomes the engine of the live 8-stock system.
* **KILL** (they don't) → we stop, having spent a small harness instead of a full build.

This is the eval harness first. Live stock-picking, position sizing (Kelly), the
15-day cadence, and the Bull/Critic/Reconciler LLM debate are OUT OF SCOPE here
(see §11).

## 2. The non-negotiables (the whole result depends on these)

1. **NO PEEKING.** Every datum carries a `knowledge_date` (when it became
   public). All data access goes through one function that filters
   `knowledge_date <= decision_date`. No module reads raw data directly.
   Enforced by an automated leak test, not by trust.
2. **SURVIVORSHIP-FREE.** The universe at any historical date includes companies
   since delisted/merged/bankrupt. Using today's survivors inflates returns
   1–4%/yr and can reverse conclusions. Enforced by an automated test (a
   known-delisted ticker must appear in its historical universe).
3. **TWO-ARM.** Every signal is judged on P(winner | signal fired) vs the
   unconditional base rate, with the losers (same-signal non-winners) explicitly
   counted. A winners-only view is forbidden.

## 3. The question, made measurable

* **Entry:** Jan 1 of each year, 2016–2026.
* **Winner** = beat the Nasdaq Composite total return by ≥15 percentage points
  over the next 12 months.
* **Universe** = Nasdaq Composite, healthcare + technology, all cap tiers,
  liquidity-filtered.
* **Edge accepted** only if the signal/archetype's conditional win-rate lift over
  base rate has a confidence interval excluding zero, survives transaction costs
  + a 1–4pp survivorship haircut, and holds in ≥3 distinct years.

## 4. Signals under test (locked)

Core (each independently replicated across decades):

* **Price momentum** (12-1 month relative strength; above a rising 200-DMA;
  52-wk-high proximity)
* **Earnings-estimate revision breadth** (net % of analysts raising)
* **Insider cluster buying** (≥3 opportunistic buyers, non-routine)
* **Quality / profitability** (Piotroski F-score, gross profitability) — used as
  a failure filter

Macro gate (AND):

* **Risk-off → cash**, when HY OAS widening + Fed tightening hard + VIX regime
  breaking

Boosters (tie-breakers, less proven):

* **Revenue acceleration** (growth rate rising, in dollars)
* **Rule of 40** (software)

## 5. The 4-agent architecture (each earns its place)

Orchestrated by a plain Python `run.py` — not an LLM (an LLM orchestrator adds
cost + non-determinism to a numeric experiment).

1. **Data-Integrity agent** — pulls survivorship-free point-in-time data; runs
   fail-loud schema/freshness/coverage checks; quarantines bad rows naming the
   exact vendor/ticker/field. (Solves the #1 historical pain: silent data
   failures.)
2. **Signal-Compute agent** — computes the locked signals + macro gate,
   deterministically. Pure, testable math.
3. **Backtest/Stats agent** — labels winners, builds the two-arm comparison,
   computes conditional win-rate lift, confidence intervals, p-values, rank-IC,
   walk-forward across years. This is the experiment.
4. **Reviewer agent** (ported `pr-reviewer`) — enforces the eval gates on every
   commit.

Deliberately excluded: Bull/Critic/Reconciler debate agents — they belong to
live picking and would bias a significance test. Added later only if the harness
proves the signals.

## 6. The data-access contract (the chokepoint)

```
get_data(field: str, ticker: str, as_of: date) -> value | None
```

* Returns only rows where `knowledge_date <= as_of`.
* Single source of truth; every gate/signal calls it; nothing bypasses it.
* `knowledge_date` is distinct from `event_date` (the period a datum describes).
  Filing lags applied: 10-Q ~40d, Form 4 +2d, 13F +45d, news/estimates at
  release timestamp.

## 7. Repo layout

```
stockscope/
├── data/          # source adapters (Sharadar default; EDGAR fallback)
├── store/         # point-in-time store + get_data() chokepoint
├── universe/      # survivorship-free universe + liquidity filter
├── signals/       # momentum, revisions, insiders, quality, boosters
├── macro/         # HY OAS, VIX, Fed-direction -> regime gate
├── backtest/      # walk-forward + purge/embargo runner
├── stats/         # two-arm lift, CIs, p-values, rank-IC
├── evals/         # tool-level + IO-level eval suites
├── report/        # per-signal lift table + GO/KILL verdict
├── tests/         # golden-case fixtures
├── .claude/agents/pr-reviewer.md   # ported PR agent
├── CLAUDE.md      # ported PR-workflow rules
└── run.py         # plain-Python orchestrator
```

## 8. Eval layers (built BEFORE the logic they guard)

* **Tool-level:** no-peek test (zero future rows); survivorship test
  (delisted ticker present in historical universe); filing-lag test; signal
  golden-cases (known input -> known output to the decimal); stats calibration
  (synthetic data with a known injected lift -> harness must recover it).
* **IO-level:** Pandera schemas (`strict=True`, `lazy=True`) on every dataframe;
  range asserts (sub-scores within bounds, win-rate in [0,1], CI bounds
  ordered); reconciliation (universe-in = winners + losers + excluded, nothing
  silently dropped); sample-size floor flag (~300/arm for significance).
* **Calibration (the validity test):** higher composite score -> higher forward
  return (decile monotonicity + rank-IC).

## 9. PR sequence (one concern each, ≤300 lines)

1. Scaffold + this ARCHITECTURE.md + README + ported pr-reviewer/CLAUDE.md + CI + empty stubs ← **current**
2. Data layer: point-in-time store + `get_data()` chokepoint + data-source adapter
3. Data-integrity evals (no-peek, survivorship, filing-lag) + Pandera schemas
4. Universe + liquidity filter
5. Signal: momentum (+ golden case)
6. Signal: estimate-revision breadth (+ golden case)
7. Signal: insider cluster buys (+ golden case)
8. Signal: quality/profitability (+ golden case)
9. Macro gate + regime classification (+ eval)
10. Winner labeler (+ golden case)
11. Two-arm stats engine + calibration eval
12. Walk-forward + purge/embargo runner
13. Report: per-signal lift table (CI/p-value) + ≥3-year consistency + GO/KILL verdict

## 10. Success / kill thresholds

* **GO:** lift CI excludes zero, survives costs + survivorship haircut, holds
  ≥3 years across ≥2 archetypes.
* **KILL:** lift not significant in ≥3 years -> abandon the concentrated-8
  thesis; default to a factor-ETF sleeve.
* **Partial:** only momentum/revisions confirm -> narrow the build to those,
  drop the rest.

## 11. Out of scope (this repo)

Live trading, order execution, Kelly position sizing, the Bull/Critic/Reconciler
LLM debate, the 15-day live cadence. These come after a GO verdict, in a
separate build.
