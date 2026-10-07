"""М2 – звернення до внутрішнього JSON-API сторінки, минаючи HTML."""
from __future__ import annotations

import httpx

from ..domain import CollectResult, ICollector, PageSpec, ProductRecord
from .common import DEFAULT_HEADERS, detect_antibot, dig, parse_price


class InternalApiCollector(ICollector):
    name = "M2"

    def is_applicable(self, page: PageSpec) -> bool:
        return page.api is not None

    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        res = CollectResult()
        api = page.api
        if api is None:
            res.error_type, res.error_msg = "not_applicable", "endpoint не задано"
            return res
        headers = {**DEFAULT_HEADERS, "Accept": "application/json", "Referer": page.url,
                   **page.headers, **api.headers}
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
                r = client.request(api.method, api.url, params=api.params or None, json=api.json_body)
            res.status_code = r.status_code
            res.bytes_wire = r.num_bytes_downloaded
            res.bytes_body = len(r.content)
            res.n_requests = res.n_xhr = 1 + len(r.history)
            res.antibot_hits, err = detect_antibot(r.status_code, r.text)
            if err:
                res.error_type = err
                return res
            if r.status_code >= 400:
                res.error_type, res.error_msg = "other", f"HTTP {r.status_code}"
                return res
            try:
                items = dig(r.json(), api.items_path) or []
                for i, it in enumerate(items, start=1):
                    name = dig(it, api.name_field)
                    price_val = dig(it, api.price_field)
                    if not name:
                        continue
                    price = float(price_val) if isinstance(price_val, (int, float)) else parse_price(str(price_val or ""))
                    res.records.append(ProductRecord(name=" ".join(str(name).split()), price=price,
                                                     price_raw=str(price_val), position=i))
            except Exception as e:  # noqa: BLE001
                res.error_type, res.error_msg = "parse", repr(e)
        except httpx.TimeoutException as e:
            res.error_type, res.error_msg = "timeout", repr(e)
        except httpx.TransportError as e:
            res.error_type, res.error_msg = "network", repr(e)
        return res
