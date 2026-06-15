"""Daily-close price adapter (the first source feeding the store).

``fetch_prices`` pulls daily closes from yfinance, falling back to Stooq, and
maps each trading day to the universal point-in-time record. For a price bar the
data is public the day it prints, so ``event_date == knowledge_date ==`` the
trading date (no filing lag — contrast §6's filing-lag sources, added later).

Fail-loud (CLAUDE.md): a failed or empty pull from BOTH providers raises
``DataPullError`` and logs which ticker/range failed. It never returns an empty
frame silently — an empty result downstream would quietly collapse the universe.
"""

from __future__ import annotations

import logging
import time
from io import StringIO

import pandas as pd
import requests

from store.schema import COLUMNS

log = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds; doubled each retry (1, 2, 4, ...)
DEFAULT_THROTTLE = 0.0  # polite pause between provider attempts


class DataPullError(RuntimeError):
    """Raised when no price data could be pulled from any provider.

    Carries ``quarantine`` (provider -> reason) so the caller / coverage report
    can record exactly why each source failed instead of silently dropping.
    """

    def __init__(self, message: str, quarantine: dict | None = None):
        super().__init__(message)
        self.quarantine = quarantine or {}


def fetch_prices(
    ticker: str,
    start,
    end,
    *,
    retries: int = DEFAULT_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    throttle: float = DEFAULT_THROTTLE,
    sleep=time.sleep,
) -> pd.DataFrame:
    """Return daily closes for ``ticker`` in ``[start, end]`` as universal records.

    Each provider is attempted with retry + exponential backoff; a polite
    ``throttle`` pause separates providers. If ALL providers exhaust their
    retries, raises ``DataPullError`` whose ``quarantine`` names why each failed
    — never an empty frame (CLAUDE.md: fail loud, no silent drops). ``sleep`` is
    injectable so tests don't actually wait.
    """
    quarantine: dict[str, str] = {}
    for label, fn in (("yfinance", _from_yfinance), ("stooq", _from_stooq)):
        frame = _pull_with_retry(
            label, fn, ticker, start, end, retries=retries,
            base_delay=base_delay, sleep=sleep,
        )
        if frame is not None and not frame.empty:
            return _to_records(frame, ticker, label)
        quarantine[label] = f"retries exhausted ({retries}) — failed or empty"
        if throttle:
            sleep(throttle)
    raise DataPullError(
        f"No price data for {ticker!r} in [{start}, {end}]; "
        f"quarantined providers: {quarantine}",
        quarantine=quarantine,
    )


def _pull_with_retry(label, fn, ticker, start, end, *, retries, base_delay, sleep):
    """Attempt ``fn`` up to ``retries`` times with exponential backoff; return a
    non-empty frame or ``None`` once exhausted. Never raises — caller decides."""
    for attempt in range(retries):
        try:
            frame, _ = fn(ticker, start, end)
        except Exception as exc:  # transport/parse error -> retry
            log.warning("%s attempt %d/%d for %s raised: %s",
                        label, attempt + 1, retries, ticker, exc)
            frame = None
        if frame is not None and not frame.empty:
            return frame
        if attempt < retries - 1:
            delay = base_delay * (2 ** attempt)
            log.warning("%s attempt %d/%d for %s failed/empty; backoff %.2fs",
                        label, attempt + 1, retries, ticker, delay)
            sleep(delay)
    return None


def _from_yfinance(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; skipping primary provider")
        return None, "yfinance"
    try:
        data = yf.download(
            ticker, start=str(start), end=str(end), progress=False, auto_adjust=False
        )
    except Exception as exc:  # network/parse errors -> fall back, don't crash
        log.warning("yfinance pull failed for %s: %s", ticker, exc)
        return None, "yfinance"
    if data is None or data.empty or "Close" not in data:
        return None, "yfinance"
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # group_by-ticker shape
        close = close.iloc[:, 0]
    return pd.DataFrame({"date": close.index, "close": close.to_numpy()}), "yfinance"


# Stooq codes indices differently from US equities (no ".us" suffix). yfinance
# uses ^-prefixed symbols (e.g. ^IXIC) directly, but Stooq needs its own code.
STOOQ_INDEX_SYMBOLS = {
    "^IXIC": "^ndq",   # Nasdaq Composite
    "^GSPC": "^spx",   # S&P 500
    "^DJI": "^dji",    # Dow Jones Industrial Average
}


def _stooq_symbol(ticker: str) -> str:
    """Map a ticker to its Stooq query symbol: index codes for known indices,
    a best-effort lowercase for other ^-symbols, else the US-equity ``.us`` form."""
    if ticker in STOOQ_INDEX_SYMBOLS:
        return STOOQ_INDEX_SYMBOLS[ticker]
    if ticker.startswith("^"):
        return ticker.lower()
    return f"{ticker.lower()}.us"


def _from_stooq(ticker, start, end):
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(ticker)}&d1={d1}&d2={d2}&i=d"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("stooq pull failed for %s: %s", ticker, exc)
        return None, "stooq"
    text = resp.text or ""
    if not text.startswith("Date,"):  # stooq emits "No data" / HTML on failure
        return None, "stooq"
    df = pd.read_csv(StringIO(text))
    if "Close" not in df.columns or df.empty:
        return None, "stooq"
    return df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"}), "stooq"


def _to_records(frame: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    ts = pd.to_datetime(frame["date"]).dt.normalize()
    rec = pd.DataFrame(
        {
            "ticker": ticker.upper(),
            "field": "close",
            "value": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            "event_date": ts.to_numpy(),
            "knowledge_date": ts.to_numpy(),
            "source": source,
        }
    ).dropna(subset=["value"])
    if rec.empty:
        raise DataPullError(f"All price rows for {ticker!r} were unparseable/empty")
    return rec[COLUMNS].reset_index(drop=True)
