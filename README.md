# stockscope — Winner-Signal Backtest Harness

A point-in-time, survivorship-free, two-arm backtest harness that decides
**GO / KILL** on a small set of proven equity signals before any live
capital or live system is built.

> Do a small set of proven signals separate future winners from losers in US
> Nasdaq healthcare + technology stocks — with a statistically significant edge
> over the base rate, holding across ≥3 distinct years?

`ARCHITECTURE.md` is the source of truth for the design, the non-negotiables,
and the PR sequence. Read it first.

## The three non-negotiables

1. **No peeking** — all data access goes through `store.get_data()`, which
   filters `knowledge_date <= as_of`. Enforced by a leak test.
2. **Survivorship-free** — the historical universe includes delisted/merged/
   bankrupt names. Enforced by a test.
3. **Two-arm** — every signal reports P(winner | signal) vs base rate, with the
   losers explicitly counted. Winners-only views are forbidden.

## Status

PR 1 — **scaffold**. Directory tree, docs, the ported `pr-reviewer` agent, CI,
and empty stubs are in place. Every module raises `NotImplementedError` until
its dedicated PR (see `ARCHITECTURE.md` §9) fills it in, evals-first.

## Repo layout

```
data/      source adapters (Sharadar default; EDGAR fallback)
store/     point-in-time store + get_data() chokepoint
universe/  survivorship-free universe + liquidity filter
signals/   momentum, revisions, insiders, quality, boosters
macro/     HY OAS, VIX, Fed-direction -> regime gate
backtest/  walk-forward + purge/embargo runner
stats/     two-arm lift, CIs, p-values, rank-IC
evals/     tool-level + IO-level eval suites
report/    per-signal lift table + GO/KILL verdict
tests/     golden-case fixtures
run.py     plain-Python orchestrator
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                 # eval + golden-case suites
python run.py --help   # orchestrator entry point
```

## Eval-first discipline

Per `ARCHITECTURE.md` §8, the eval that guards a piece of logic lands **before
or with** that logic — never after. Tool-level evals (no-peek, survivorship,
filing-lag, signal golden cases, stats calibration) and IO-level evals (Pandera
schemas, range asserts, reconciliation, sample-size floor) are first-class.

## Out of scope

Live trading, order execution, Kelly sizing, the Bull/Critic/Reconciler LLM
debate, and the 15-day live cadence are explicitly **not** in this repo
(`ARCHITECTURE.md` §11). They come after a GO verdict, in a separate build.
