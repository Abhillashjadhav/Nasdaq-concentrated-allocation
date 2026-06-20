"""Locked signals (ARCHITECTURE.md §4) and boosters.

Core: momentum, estimate-revision breadth, quality. (Insider cluster buys were
dropped when the EDGAR/Form 4 source was removed — SimFin is now the sole
fundamentals + sector source.) Boosters: revenue acceleration, Rule of 40. Every
signal is pure, deterministic math and reads data ONLY through ``store.get_data``.
The signal set is locked — adding/swapping one is a design change, not a PR.
"""
