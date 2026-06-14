# CLAUDE.md — stockscope PR workflow rules

`ARCHITECTURE.md` is the source of truth. These rules govern how changes land.
The `pr-reviewer` agent (`.claude/agents/pr-reviewer.md`) enforces them on every PR.

## The three non-negotiables (never violate)

1. **No peeking.** No module reads raw data directly. All data access goes
   through `store.get_data(field, ticker, as_of)`, which returns only rows with
   `knowledge_date <= as_of`. Any new data read that bypasses the chokepoint is
   a hard reject.
2. **Survivorship-free.** The historical universe must include delisted/merged/
   bankrupt names. Never filter the universe to today's survivors. The
   survivorship test (a known-delisted ticker present in its historical
   universe) must pass.
3. **Two-arm only.** Every signal result reports P(winner | signal fired) vs the
   unconditional base rate, with same-signal losers explicitly counted. A
   winners-only view is forbidden.

## How work is sequenced

* Follow the PR sequence in `ARCHITECTURE.md` §9. One concern per PR.
* **≤300 lines** of meaningful change per PR (scaffolding/fixtures excepted).
* **Evals before logic.** The eval that guards a piece of logic lands before or
  with it, never after (`ARCHITECTURE.md` §8).
* The signals under test are **locked** (`ARCHITECTURE.md` §4). Adding or
  swapping a signal is a design change, not a PR — raise it explicitly.

## Engineering rules

* Determinism: signal/stats math is pure and seedable. No wall-clock, no network
  inside compute. The orchestrator is plain Python, never an LLM.
* Fail loud: data-integrity checks quarantine bad rows naming the exact
  vendor/ticker/field. Empty/collapsed outputs are surfaced, never silently
  passed.
* Schemas: every dataframe crossing a module boundary is validated with a
  Pandera schema (`strict=True`, `lazy=True`).
* `knowledge_date` ≠ `event_date`. Filing lags (10-Q ~40d, Form 4 +2d, 13F +45d,
  news/estimates at release) are applied at ingest, not in signal code.

## Honesty rails

* No look-ahead, ever — not "just for a quick check". A leaked future row
  invalidates the whole experiment.
* No result inflation: report the conditional lift and its CI as computed.
  Survives-costs and survivorship-haircut adjustments are shown, not hidden.
* If a stage produces an implausible (empty/collapsed) output, halt and surface
  it rather than carrying it downstream.

## Secrets

* No credentials, tokens, or API keys in the repo — ever. Read them from the
  environment (`.env` is gitignored). A committed secret is a hard reject and
  must be rotated, not just deleted.

## What NOT to do

* Don't bypass `get_data()`.
* Don't use today's survivor universe for a historical date.
* Don't present a winners-only result.
* Don't add logic without its guarding eval.
* Don't pull live-trading / Kelly / LLM-debate concerns into this repo (§11).
