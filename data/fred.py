"""FRED macro adapter (ARCHITECTURE.md §5, §6).

``fetch_macro`` ingests the three macro series the regime gate needs — HY OAS,
Fed funds rate, VIX — from the St. Louis Fed (FRED) into the store under the
sentinel ``MACRO`` ticker, mapping each observation to a universal point-in-time
record and writing it via the existing ``store.put_data`` (Pandera + fail-loud).

Point-in-time (§6): ``event_date`` = the observation date; ``knowledge_date`` =
the observation date plus a documented per-series PUBLICATION LAG. FRED's basic
observations endpoint returns (date, value) but not the per-point release date,
so we apply a conservative fixed lag (default 1 day — these daily series publish
~next business day) so the gate can never read a not-yet-released print. Raise the
lag for a series if its true release lag is larger; a too-small lag would be a
look-ahead.

The FRED API key is configuration (free), read from ``STOCKSCOPE_FRED_API_KEY``
(or passed in); the client refuses to run if it is unset. The retry/backoff +
throttle pattern mirrors ``data/edgar_client.py``.

Missing values (FRED's ``"."`` sentinel) are skipped; a series with no usable
observations is FLAGGED as a coverage gap, never written as zero. Field names
match ``macro.regime`` (guarded by a test).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field as dc_field

import pandas as pd
import requests

import store as store_pkg
from store.schema import COLUMNS

DEFAULT_BASE_URL = "https://api.stlouisfed.org/fred"
API_KEY_ENV = "STOCKSCOPE_FRED_API_KEY"
MIN_INTERVAL_SECONDS = 0.1  # be polite to FRED
DEFAULT_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
SOURCE = "fred"
MACRO_TICKER = "MACRO"

# field -> (FRED series id, publication lag in BUSINESS days). Field names match
# macro.regime. A business-day lag avoids exposing a Friday print on Sat/Sun
# before its actual next-business-day release (holidays are a future refinement).
# Note: this ingests FRED's latest vintage, not ALFRED point-in-time vintages, so
# later revisions are not modelled — acceptable for the coarse regime gate.
SERIES = {
    "hy_oas": ("BAMLH0A0HYM2", 1),       # ICE BofA US High Yield OAS (daily)
    "fed_funds_rate": ("DFF", 1),        # daily effective federal funds rate
    "vix": ("VIXCLS", 1),                # CBOE VIX close (daily)
}


class FredConfigError(RuntimeError):
    """Raised when the required FRED API key is missing."""


class FredHTTPError(RuntimeError):
    """Raised when a FRED request fails after exhausting retries."""


class FredClient:
    """Throttled, retrying JSON client for the FRED observations API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        retries: int = DEFAULT_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        min_interval: float = MIN_INTERVAL_SECONDS,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not key:
            raise FredConfigError(
                f"FRED API key required; set {API_KEY_ENV} (free at "
                f"https://fred.stlouisfed.org/docs/api/api_key.html)."
            )
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.base_delay = base_delay
        self.min_interval = min_interval
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get_observations(self, series_id: str, *, observation_start: str | None = None):
        """Return the FRED observations JSON for ``series_id`` (retried/throttled)."""
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        if observation_start:
            params["observation_start"] = observation_start
        last_exc = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(
                    f"{self.base_url}/series/observations", params=params,
                    headers={"User-Agent": "stockscope"}, timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # transport/HTTP/parse -> retry then fail loud
                last_exc = exc
                if attempt < self.retries - 1:
                    self._sleep(self.base_delay * (2 ** attempt))
        # The api_key rides in the query string, so the underlying error (and its
        # URL) can carry it. Redact the key and suppress the chained cause so it
        # never reaches logs/tracebacks (CLAUDE.md: a logged secret is a hard reject).
        detail = str(last_exc).replace(self.api_key, "***")
        status = getattr(getattr(last_exc, "response", None), "status_code", None)
        raise FredHTTPError(
            f"FRED request failed for {series_id} (status={status}): {detail}"
        ) from None


@dataclass
class FredMacroResult:
    records: pd.DataFrame
    gaps: list[dict] = dc_field(default_factory=list)
    n_written: int = 0


def fetch_macro(
    *,
    client: FredClient | None = None,
    store=None,
    write: bool = False,
    fields: list[str] | None = None,
    observation_start: str | None = None,
) -> FredMacroResult:
    """Pull the macro series from FRED into universal records. Optionally write to
    the store. A series with no usable observations is flagged, never zeroed."""
    client = client or FredClient()
    fields = fields or list(SERIES)

    rows: list[dict] = []
    gaps: list[dict] = []
    for name in fields:
        series_id, lag_bdays = SERIES[name]
        data = client.get_observations(series_id, observation_start=observation_start)
        usable = [
            o for o in data.get("observations", [])
            if o.get("date") and o.get("value") not in (None, ".", "")
        ]
        if not usable:
            gaps.append({"ticker": MACRO_TICKER, "field": name,
                         "reason": "no_fred_observations", "vendor": SOURCE})
            continue
        for o in usable:
            event_date = pd.Timestamp(o["date"])
            rows.append({
                "ticker": MACRO_TICKER, "field": name, "value": float(o["value"]),
                "event_date": event_date,
                "knowledge_date": event_date + pd.offsets.BusinessDay(lag_bdays),
                "source": SOURCE,
            })

    records = pd.DataFrame(rows, columns=COLUMNS)
    n_written = 0
    if write and not records.empty:
        n_written = (store or store_pkg).put_data(records)
    return FredMacroResult(records, gaps, n_written)
