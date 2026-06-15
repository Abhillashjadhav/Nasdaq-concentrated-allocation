"""Nasdaq Trader symbol-directory adapter (the free US-listed symbol set).

Pulls the current US-listed common-stock symbols from Nasdaq Trader's free
symbol-directory files (``nasdaqlisted.txt`` + ``otherlisted.txt``), with retry
and a simple on-disk cache. Test issues and ETFs are dropped. This is the
candidate symbol set the universe builder classifies by SIC.

Note (the $0 gap / survivorship): these files list TODAY's listed names — they do
not contain already-delisted companies. So a universe built from them is
SURVIVOR-LIMITED. We never silently call that survivorship-free (ARCHITECTURE.md
§2.2); the limitation is documented here and surfaced in the report.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
DEFAULT_CACHE = Path(".data_cache/nasdaq_symbols.txt")
_RETRIES = 3
_BASE_DELAY = 1.0


def _http_get(url, *, retries=_RETRIES, base_delay=_BASE_DELAY, sleep=time.sleep) -> str:
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # transport -> retry then fail loud
            last = exc
            if attempt < retries - 1:
                sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"failed to fetch {url}: {last}") from last


def _parse_pipe_table(text: str, symbol_col_names) -> list[str]:
    """Parse a pipe-delimited Nasdaq directory file, dropping test issues/ETFs and
    the trailing 'File Creation Time' line. Returns the symbol column values."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")

    def col(name):
        return header.index(name) if name in header else None

    sym_i = next((col(n) for n in symbol_col_names if col(n) is not None), 0)
    test_i, etf_i = col("Test Issue"), col("ETF")
    out = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            continue
        f = ln.split("|")
        if test_i is not None and test_i < len(f) and f[test_i].strip() == "Y":
            continue
        if etf_i is not None and etf_i < len(f) and f[etf_i].strip() == "Y":
            continue
        sym = f[sym_i].strip()
        if sym and sym.isascii() and all(c.isalnum() or c in ".-" for c in sym):
            out.append(sym)
    return out


def fetch_listed_symbols(*, http_get=None, cache_path: Path | None = DEFAULT_CACHE,
                         use_cache: bool = True) -> list[str]:
    """Return the sorted, de-duped set of US-listed symbols (cached on disk).
    ``http_get`` is injectable for tests."""
    cache_path = Path(cache_path) if cache_path else None
    if use_cache and cache_path and cache_path.exists():
        return sorted({s for s in cache_path.read_text().split() if s})

    get = http_get or _http_get
    symbols = set()
    symbols.update(_parse_pipe_table(get(NASDAQ_LISTED_URL), ["Symbol"]))
    symbols.update(_parse_pipe_table(get(OTHER_LISTED_URL), ["ACT Symbol", "NASDAQ Symbol"]))
    result = sorted(symbols)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(result))
    return result
