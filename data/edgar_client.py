"""Thin HTTP client for SEC EDGAR (data.sec.gov) + CIK resolver.

ARCHITECTURE.md §5 (Data-Integrity agent) / §6: this is the EDGAR fallback source
deferred from the data-layer PR. It is the only place that talks to SEC over the
network; parsed facts flow into the store via ``put_data`` so the no-peek and
fail-loud contracts still hold.

SEC requires a descriptive ``User-Agent`` carrying contact info (a name + email);
without it EDGAR returns HTTP 403. It is configuration, NOT a secret, so it is
read from the ``STOCKSCOPE_SEC_USER_AGENT`` environment variable (or passed
explicitly) and the client refuses to run if it is unset — better a loud config
error than silent 403s. Requests are throttled to <=10/s (SEC's fair-access
limit) and use the same retry/backoff pattern as ``data/prices.py``.
"""

from __future__ import annotations

import os
import time

import requests

DEFAULT_BASE_URL = "https://data.sec.gov"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT_ENV = "STOCKSCOPE_SEC_USER_AGENT"
MIN_INTERVAL_SECONDS = 0.1  # <=10 requests/second
DEFAULT_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0


class EdgarConfigError(RuntimeError):
    """Raised when required configuration (the SEC User-Agent) is missing."""


class EdgarHTTPError(RuntimeError):
    """Raised when an EDGAR request fails after exhausting retries."""


class UnknownTickerError(KeyError):
    """Raised when a ticker is not present in SEC's company_tickers.json."""


class EdgarClient:
    """Throttled, retrying JSON client for SEC EDGAR."""

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        retries: int = DEFAULT_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        min_interval: float = MIN_INTERVAL_SECONDS,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        ua = (user_agent or os.environ.get(USER_AGENT_ENV) or "").strip()
        if not ua:
            raise EdgarConfigError(
                f"SEC requires a User-Agent with contact info; set {USER_AGENT_ENV} "
                f"to e.g. 'Jane Doe jane@example.com'. EDGAR returns 403 without it."
            )
        self.user_agent = ua
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

    def get_json(self, path_or_url: str):
        """GET JSON from an EDGAR path (or absolute URL), throttled and retried.
        Raises ``EdgarHTTPError`` after exhausting retries."""
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        last_exc = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # transport/HTTP/parse -> retry then fail loud
                last_exc = exc
                if attempt < self.retries - 1:
                    self._sleep(self.base_delay * (2 ** attempt))
        raise EdgarHTTPError(f"EDGAR request failed for {url}: {last_exc}") from last_exc


class CikResolver:
    """Resolve ticker -> zero-padded 10-digit CIK from SEC's company_tickers.json,
    fetched once and cached. Unknown tickers fail loud (never guessed)."""

    def __init__(self, client: EdgarClient, *, url: str = COMPANY_TICKERS_URL):
        self.client = client
        self.url = url
        self._map: dict[str, str] | None = None

    def _ensure_loaded(self) -> dict[str, str]:
        if self._map is None:
            data = self.client.get_json(self.url)
            self._map = {
                str(row["ticker"]).upper(): f"{int(row['cik_str']):010d}"
                for row in data.values()
            }
        return self._map

    def resolve(self, ticker: str) -> str:
        cik = self._ensure_loaded().get(str(ticker).upper())
        if cik is None:
            raise UnknownTickerError(
                f"ticker {ticker!r} not found in SEC company_tickers.json"
            )
        return cik
