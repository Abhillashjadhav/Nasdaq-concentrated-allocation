"""Nasdaq symbol-directory parsing (offline)."""

from __future__ import annotations

from universe.nasdaq_directory import fetch_listed_symbols

NASDAQ_TXT = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "TSTX|Test Issue|Q|Y|N|100|N|N\n"            # test issue -> dropped
    "QQQ|Invesco QQQ Trust|Q|N|N|100|Y|N\n"      # ETF -> dropped
    "File Creation Time: 0615202509:30|||||||\n"
)
OTHER_TXT = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "IBM|Intl Business Machines|N|IBM|N|100|N|IBM\n"
    "SPY|SPDR S&P 500|P|SPY|Y|100|N|SPY\n"       # ETF -> dropped
    "File Creation Time: 0615202509:30||||||||\n"
)


def _fake_get(url, **kwargs):
    return NASDAQ_TXT if "nasdaqlisted" in url else OTHER_TXT


def test_parse_drops_test_issues_and_etfs():
    syms = fetch_listed_symbols(http_get=_fake_get, cache_path=None)
    assert "AAPL" in syms and "IBM" in syms        # real common stock kept
    assert "TSTX" not in syms                       # test issue dropped
    assert "QQQ" not in syms and "SPY" not in syms  # ETFs dropped
    assert all("File Creation Time" not in s for s in syms)  # trailer line ignored
    assert syms == sorted(set(syms))                # sorted + de-duped
