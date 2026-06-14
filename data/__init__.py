"""Source adapters feeding the point-in-time store.

Sharadar is the default vendor; EDGAR is the fallback. Adapters are the ONLY
place that touches a vendor: they pull survivorship-free point-in-time rows,
attach a ``knowledge_date`` (applying filing lags), and hand them to the store.
Downstream code never imports an adapter directly — it calls ``store.get_data``.

Implemented in PR 2 (ARCHITECTURE.md §9).
"""
