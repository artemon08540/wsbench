"""М4 – гібридний метод: спочатку дешевий М1, і лише якщо він зібрав замало – М3.

Правило перемикання (поріг фіксується ДО основного експерименту за пілотом):
  запускати М3, якщо  n_M1 < threshold * expected_count  або  n_M1 < min_records.
Якщо expected_count невідомий – діє лише умова min_records.
"""
from __future__ import annotations

from ..domain import CollectResult, ICollector, PageSpec
from .headless import HeadlessCollector
from .http_html import HttpHtmlCollector


class HybridCollector(ICollector):
    name = "M4"

    def __init__(self, threshold: float = 0.8, min_records: int = 5, headless: bool = True):
        self.threshold = threshold
        self.min_records = min_records
        self.m1 = HttpHtmlCollector()
        self.m3 = HeadlessCollector(headless=headless)

    def needs_browser(self, n_found: int, page: PageSpec) -> bool:
        if n_found < self.min_records:
            return True
        return bool(page.expected_count) and n_found < self.threshold * page.expected_count

    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        first = self.m1.collect(page, timeout_s)
        n1 = len(first.records)
        if first.error_type in ("http_403", "http_429", "captcha") or not self.needs_browser(n1, page):
            first.extra.update(fallback=False, m1_found=n1, threshold=self.threshold)
            return first

        second = self.m3.collect(page, timeout_s)
        # Ресурси обох кроків сумуються: гібрид платить і за марну спробу М1.
        second.bytes_wire += first.bytes_wire
        second.bytes_body += first.bytes_body
        second.n_requests += first.n_requests
        second.antibot_hits += first.antibot_hits
        second.html_size = first.html_size
        second.extra.update(fallback=True, m1_found=n1, threshold=self.threshold)
        return second
