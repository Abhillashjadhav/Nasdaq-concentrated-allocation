"""Hard no-peek leak check (ARCHITECTURE.md §2, §8).

The unit test in ``tests/test_store.py`` checks the chokepoint at one boundary.
This is the *hard* guard: it fuzzes many (field, ticker, as_of) combinations
against a populated store and asserts that NO call ever returns a row whose
``knowledge_date`` is later than the requested ``as_of``. A single leaked future
row invalidates the whole experiment, so on any leak this raises ``LeakError``
naming the exact ticker/field/knowledge_date/as_of — and the build fails.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


class LeakError(AssertionError):
    """Raised when get_data returns a row known after the requested as_of."""


def assert_no_future_rows(store, fields, tickers, as_of_dates) -> int:
    """Call ``store.get_data`` across the cartesian product of inputs and assert
    every returned row satisfies ``knowledge_date <= as_of``.

    Returns the number of (field, ticker, as_of) probes performed. Raises
    ``LeakError`` on the first leaked row, naming the exact offender.
    """
    probes = 0
    for as_of in as_of_dates:
        cutoff = pd.Timestamp(as_of).normalize()
        for field in fields:
            for ticker in tickers:
                rows = store.get_data(field, ticker, as_of)
                probes += 1
                if rows.empty:
                    continue
                leaked = rows[pd.to_datetime(rows["knowledge_date"]) > cutoff]
                if not leaked.empty:
                    bad = leaked.iloc[0]
                    raise LeakError(
                        f"LEAK: get_data({field!r}, {ticker!r}, as_of={as_of}) "
                        f"returned a row with knowledge_date="
                        f"{pd.Timestamp(bad['knowledge_date']).date()} "
                        f"(> as_of); source={bad['source']!r}"
                    )
    return probes
