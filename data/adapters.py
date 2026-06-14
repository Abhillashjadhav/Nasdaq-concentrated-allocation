"""Vendor source adapters (Sharadar default; EDGAR fallback).

Each adapter returns point-in-time rows with an attached ``knowledge_date`` so
the store can enforce the no-peek contract. Filing lags (10-Q ~40d, Form 4 +2d,
13F +45d, news/estimates at release) are applied HERE, at ingest.

Stub — implemented in PR 2 (ARCHITECTURE.md §9).
"""

from __future__ import annotations


def load_sharadar(*args, **kwargs):
    """Pull survivorship-free point-in-time rows from Sharadar."""
    raise NotImplementedError("Sharadar adapter lands in PR 2")


def load_edgar(*args, **kwargs):
    """Fallback adapter: pull filings from SEC EDGAR."""
    raise NotImplementedError("EDGAR fallback adapter lands in PR 2")
