"""Реєстр збирачів. Щоб додати п'ятий метод – реалізуйте ICollector і додайте його сюди;
раннер і аналітику змінювати не потрібно."""
from ..domain import ICollector
from .headless import HeadlessCollector
from .http_html import HttpHtmlCollector
from .hybrid import HybridCollector
from .internal_api import InternalApiCollector


def build_collectors(names: list[str], hybrid_threshold: float = 0.8, headless: bool = True) -> dict[str, ICollector]:
    factory = {
        "M1": lambda: HttpHtmlCollector(),
        "M2": lambda: InternalApiCollector(),
        "M3": lambda: HeadlessCollector(headless=headless),
        "M4": lambda: HybridCollector(threshold=hybrid_threshold, headless=headless),
    }
    unknown = [n for n in names if n not in factory]
    if unknown:
        raise ValueError(f"Невідомі методи: {unknown}. Доступні: {list(factory)}")
    return {n: factory[n]() for n in names}


__all__ = ["build_collectors", "HttpHtmlCollector", "InternalApiCollector", "HeadlessCollector", "HybridCollector"]
