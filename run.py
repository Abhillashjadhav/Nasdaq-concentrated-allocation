"""stockscope orchestrator — plain Python, never an LLM (ARCHITECTURE.md §5).

Wires the 4-agent flow: Data-Integrity -> Signal-Compute -> Backtest/Stats ->
Report. Deterministic by design; no LLM in the loop of a numeric experiment.

This is the PR-1 scaffold: the CLI parses but the pipeline stages are stubs that
land in their dedicated PRs (ARCHITECTURE.md §9).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stockscope",
        description="Winner-signal backtest harness (GO/KILL). See ARCHITECTURE.md.",
    )
    parser.add_argument(
        "--start-year", type=int, default=2016, help="first entry year (Jan 1)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2026, help="last entry year (Jan 1)"
    )
    args = parser.parse_args(argv)

    raise NotImplementedError(
        f"Pipeline stages land in PRs 2–13 (ARCHITECTURE.md §9). "
        f"Requested entry years: {args.start_year}–{args.end_year}."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
