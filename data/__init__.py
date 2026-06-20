"""Source adapters feeding the point-in-time store.

Adapters are the ONLY place that touches a vendor: they pull survivorship-free
point-in-time rows, attach a ``knowledge_date`` (applying filing lags), and hand
them to the store. Downstream code never imports an adapter directly — it calls
``store.get_data``. The live adapters are ``data/prices.py`` (prices) and
``data/simfin_client.py`` (SimFin bulk fundamentals + sector reference).
"""
