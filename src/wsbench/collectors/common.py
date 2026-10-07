"""Спільні функції збирачів: розбір ціни, вилучення карток із HTML, ознаки анти-бот-захисту.

Важливо для коректності порівняння: М1 і М3 використовують ОДНУ й ту саму функцію
extract_from_html(). Методи відрізняються лише тим, ЯК отримано HTML (сирий HTTP-відповідь
чи DOM після виконання JavaScript), а не тим, як із нього вилучаються дані.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from ..domain import PageSpec, ProductRecord

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
    "Upgrade-Insecure-Requests": "1",
}

CAPTCHA_MARKERS = (
    "g-recaptcha", "hcaptcha", "cf-challenge", "challenge-platform", "cf-turnstile",
    "captcha-delivery", "px-captcha", "are you a robot", "verify you are human",
    "attention required! | cloudflare", "just a moment...",
)

_NUM_RE = re.compile(r"\d[\d\s   .,']*")


def parse_price(text: Optional[str]) -> Optional[float]:
    """'12 999 ₴' -> 12999.0; '1 299,50 грн' -> 1299.5; '$1,299.99' -> 1299.99.

    Береться ПЕРШЕ число в тексті, тому селектор ціни має вказувати на актуальну ціну,
    а не на блок, де є і стара (закреслена), і нова.
    """
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    s = re.sub(r"[\s   ']", "", m.group(0)).rstrip(".,")
    if not s:
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = (head.replace(",", "") + "." + tail) if len(tail) in (1, 2) else s.replace(",", "")
    elif "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3 and (s.count(".") > 1 or len(head) <= 3):
            s = s.replace(".", "")          # 1.299 -> 1299 (крапка як роздільник тисяч)
        else:
            s = head.replace(".", "") + "." + tail
    try:
        return float(s)
    except ValueError:
        return None


def _text(el) -> str:
    return " ".join(el.get_text(" ", strip=True).split()) if el is not None else ""


def extract_from_html(html: str, page: PageSpec) -> list[ProductRecord]:
    soup = BeautifulSoup(html, "lxml")
    records: list[ProductRecord] = []
    for i, card in enumerate(soup.select(page.card_selector), start=1):
        name = _text(card.select_one(page.name_selector))
        price_raw = _text(card.select_one(page.price_selector))
        if not name:
            continue
        records.append(ProductRecord(name=name, price=parse_price(price_raw), price_raw=price_raw, position=i))
    return records


def count_cards(html: str, card_selector: str) -> int:
    return len(BeautifulSoup(html, "lxml").select(card_selector))


def detect_antibot(status: Optional[int], body: str) -> tuple[int, Optional[str]]:
    """Повертає (кількість ознак блокування, тип помилки або None)."""
    if status == 403:
        return 1, "http_403"
    if status == 429:
        return 1, "http_429"
    low = body[:200_000].lower()
    if any(mk in low for mk in CAPTCHA_MARKERS):
        return 1, "captcha"
    return 0, None


def dig(obj: Any, path: str) -> Any:
    """Дістає значення за шляхом 'data.items.0.name' зі вкладеного JSON."""
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur
