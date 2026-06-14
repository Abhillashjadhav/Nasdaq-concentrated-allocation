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
from io import StringIO

import pandas as pd
import requests

from store.schema import COLUMNS

log = logging.getLogger(__name__)


class DataPullError(RuntimeError):
    """Raised when no price data could be pulled from any provider."""


def fetch_prices(ticker: str, start, end) -> pd.DataFrame:
    """Return daily closes for ``ticker`` in ``[start, end]`` as universal records.

    Columns match ``store.schema`` (ticker, field="close", value, event_date,
    knowledge_date, source). Raises ``DataPullError`` if both providers fail.
    """
    frame, source = _from_yfinance(ticker, start, end)
    if frame is None or frame.empty:
        log.warning("yfinance returned no rows for %s; falling back to Stooq", ticker)
        frame, source = _from_stooq(ticker, start, end)
    if frame is None or frame.empty:
        raise DataPullError(
            f"No price data for {ticker!r} in [{start}, {end}] from yfinance or stooq"
        )
    return _to_records(frame, ticker, source)


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


def _from_stooq(ticker, start, end):
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&d1={d1}&d2={d2}&i=d"
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
