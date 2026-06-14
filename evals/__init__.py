"""Eval suites — built BEFORE the logic they guard (ARCHITECTURE.md §8).

Tool-level: no-peek (zero future rows), survivorship (delisted ticker present in
its historical universe), filing-lag, signal golden cases, stats calibration
(recover a known injected lift).

IO-level: Pandera schemas (strict, lazy), range asserts, reconciliation
(universe-in = winners + losers + excluded), sample-size floor (~300/arm).

Each eval lands in the PR that introduces the logic it guards.
"""
