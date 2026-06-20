"""Vendor source adapters.

Each adapter returns point-in-time rows with an attached ``knowledge_date`` so
the store can enforce the no-peek contract. Filing lags (10-Q ~40d, 13F +45d,
news/estimates at release) are applied HERE, at ingest. The live adapters are
``data/prices.py`` (prices) and ``data/simfin_client.py`` (SimFin bulk
fundamentals + sector reference).

Stub — the Sharadar adapter is deferred (ARCHITECTURE.md §9).
"""

from __future__ import annotations


def load_sharadar(*args, **kwargs):
    """Pull survivorship-free point-in-time rows from Sharadar."""
    raise NotImplementedError("Sharadar adapter lands in PR 2")
