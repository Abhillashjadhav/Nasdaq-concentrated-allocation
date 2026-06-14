"""PR-1 scaffold smoke tests.

These assert the structure the rest of the harness is built on — the package
tree exists and the no-peek chokepoint is present with the contracted signature.
They intentionally do NOT exercise behaviour (every stage is a stub until its
PR), so CI is green from PR 1 onward without faking results.
"""

from __future__ import annotations

import importlib
import inspect
from datetime import date

import pytest

PACKAGES = [
    "data",
    "store",
    "universe",
    "signals",
    "macro",
    "backtest",
    "stats",
    "evals",
    "report",
]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_importable(pkg):
    assert importlib.import_module(pkg) is not None


def test_get_data_is_the_chokepoint():
    """The no-peek contract: store.get_data(field, ticker, as_of)."""
    store = importlib.import_module("store")
    assert hasattr(store, "get_data")
    params = list(inspect.signature(store.get_data).parameters)
    assert params == ["field", "ticker", "as_of"]


def test_stubs_raise_not_implemented():
    """Not-yet-built stages must fail loud, not return a silent value.

    (store, universe, signals, macro and the winner labeler are implemented in
    their PRs; here we guard a stage still on the stub — the walk-forward runner,
    PR 12.)
    """
    backtest = importlib.import_module("backtest")
    with pytest.raises(NotImplementedError):
        backtest.run_walk_forward()


def test_orchestrator_cli_parses():
    run = importlib.import_module("run")
    with pytest.raises(NotImplementedError):
        run.main(["--start-year", "2016", "--end-year", "2026"])
