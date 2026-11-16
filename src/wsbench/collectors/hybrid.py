"""М4 – гібридний метод: спочатку дешевий М1, і лише якщо його відповідь виглядає неповною – М3.

Критерій перемикання використовує ЛИШЕ те, що реальний збирач бачить у відповіді М1
(кількість товарів з еталону НЕ використовується – це була б підказка, недоступна на практиці):

  запускати М3, якщо виконується хоча б одна умова:
    (1) мало записів:            n_M1 < min_records;
    (2) неповні ключові поля:    fill < threshold, де fill – частка карток у HTML,
                                 у яких заповнено і назву, і ціну (напр., ціни малює JavaScript);
    (3) ознака довантаження:     у вихідному HTML присутня кнопка «Показати ще»
                                 (load_more_selector із конфігурації сторінки, яку використовує і М3).

Блокування (403/429/CAPTCHA) не є підставою для перемикання: браузер не повинен
використовуватися як засіб обходу захисту.

Поріг threshold фіксується до експерименту; чутливість до нього (0,6…0,9) оцінюється
за журналом (поле m1_fill), бо рішення є детермінованою функцією відповіді М1.

rule="oracle" – попередня версія критерію (n_M1 < threshold·expected_count), залишена лише
для відтворення результатів першого запуску; у дослідженні не використовується.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from ..domain import CollectResult, ICollector, PageSpec
from .common import extract_from_html
from .headless import HeadlessCollector
from .http_html import fetch_html


def m1_signals(html: str, page: PageSpec, records) -> dict:
    """Ознаки неповноти відповіді М1, обчислені лише з її HTML."""
    soup = BeautifulSoup(html or "", "lxml")
    n_cards = len(soup.select(page.card_selector)) if html else 0
    full = sum(1 for r in records if r.name and r.price is not None)
    fill = full / n_cards if n_cards else 0.0
    load_more = bool(page.load_more_selector and page.load_more_clicks > 0
                     and soup.select_one(page.load_more_selector) is not None)
    return {"n_cards": n_cards, "fill": round(fill, 4), "load_more": load_more}


class HybridCollector(ICollector):
    name = "M4"

    def __init__(self, threshold: float = 0.8, min_records: int = 5, headless: bool = True, rule: str = "fields"):
        self.threshold = threshold
        self.min_records = min_records
        self.rule = rule
        self.m3 = HeadlessCollector(headless=headless)

    def switch_reason(self, n_found: int, sig: dict, page: PageSpec) -> str:
        """'' – браузер не потрібен; інакше – причина перемикання."""
        if n_found < self.min_records:
            return "few_records"
        if self.rule == "oracle":
            return "below_expected" if page.expected_count and n_found < self.threshold * page.expected_count else ""
        if sig["fill"] < self.threshold:
            return "low_fill"
        if sig["load_more"]:
            return "load_more"
        return ""

    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        first, html = fetch_html(page, timeout_s)
        if html and first.error_type is None:
            try:
                first.records = extract_from_html(html, page)
            except Exception as e:  # noqa: BLE001
                first.error_type, first.error_msg = "parse", repr(e)
        n1 = len(first.records)
        sig = m1_signals(html, page, first.records) if first.error_type is None else {"n_cards": 0, "fill": 0.0, "load_more": False}
        info = dict(m1_found=n1, m1_fill=sig["fill"], m1_load_more=sig["load_more"], threshold=self.threshold, rule=self.rule)

        if first.error_type in ("http_403", "http_429", "captcha"):
            first.extra.update(fallback=False, switch_reason="blocked", **info)
            return first
        reason = self.switch_reason(n1, sig, page)
        if not reason:
            first.extra.update(fallback=False, switch_reason="", **info)
            return first

        second = self.m3.collect(page, timeout_s)
        # Ресурси обох кроків сумуються: гібрид платить і за марну спробу М1.
        second.bytes_wire += first.bytes_wire
        second.bytes_body += first.bytes_body
        second.n_requests += first.n_requests
        second.antibot_hits += first.antibot_hits
        second.html_size = first.html_size
        second.extra.update(fallback=True, switch_reason=reason, **info)
        return second
