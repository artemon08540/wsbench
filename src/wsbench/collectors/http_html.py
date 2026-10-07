"""М1 – одиночний HTTP-запит із розбором HTML (baseline). JavaScript не виконується."""
from __future__ import annotations

import httpx

from ..domain import CollectResult, ICollector, PageSpec
from .common import DEFAULT_HEADERS, detect_antibot, extract_from_html


def fetch_html(page: PageSpec, timeout_s: int) -> tuple[CollectResult, str]:
    """Один GET-запит новим клієнтом (ізоляція стану: без cookie і кешу з попередніх прогонів)."""
    res = CollectResult()
    html = ""
    headers = {**DEFAULT_HEADERS, **page.headers}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
            r = client.get(page.url)
        html = r.text
        res.status_code = r.status_code
        res.bytes_wire = r.num_bytes_downloaded
        res.bytes_body = len(r.content)
        res.n_requests = 1 + len(r.history)
        res.html_size = len(r.content)
        res.antibot_hits, err = detect_antibot(r.status_code, html)
        if err:
            res.error_type = err
        elif r.status_code >= 400:
            res.error_type, res.error_msg = "other", f"HTTP {r.status_code}"
    except httpx.TimeoutException as e:
        res.error_type, res.error_msg = "timeout", repr(e)
    except httpx.TransportError as e:
        res.error_type, res.error_msg = "network", repr(e)
    return res, html


class HttpHtmlCollector(ICollector):
    name = "M1"

    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        res, html = fetch_html(page, timeout_s)
        if html and res.error_type is None:
            try:
                res.records = extract_from_html(html, page)
            except Exception as e:  # noqa: BLE001
                res.error_type, res.error_msg = "parse", repr(e)
        return res
