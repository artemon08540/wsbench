"""Моделі даних і контракти (шар domain). Не залежить від інших модулів."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ApiSpec:
    """Опис внутрішнього JSON-API сторінки (для методу М2).

    Знаходиться вручну: DevTools -> Network -> Fetch/XHR.
    """
    url: str
    method: str = "GET"
    params: dict[str, Any] = field(default_factory=dict)
    json_body: Optional[dict[str, Any]] = None
    headers: dict[str, str] = field(default_factory=dict)
    items_path: str = ""          # шлях до списку товарів у JSON, напр. "data.products"
    name_field: str = "name"      # шлях до назви всередині елемента
    price_field: str = "price"    # шлях до ціни всередині елемента


@dataclass
class PageSpec:
    """Одна сторінка вибірки з усіма налаштуваннями збирачів."""
    page_id: str
    site: str
    url: str
    card_selector: str                       # CSS-селектор картки товару
    name_selector: str                       # CSS-селектор назви всередині картки
    price_selector: str                      # CSS-селектор актуальної ціни всередині картки
    render_type: str = "unknown"             # ssr | spa | hybrid | unknown
    pagination: str = "unknown"              # classic | load_more | infinite | unknown
    expected_count: Optional[int] = None     # очікувана кількість товарів (з еталону)
    wait_selector: Optional[str] = None      # що чекати в М3 (за замовчуванням card_selector)
    scroll: bool = False                     # чи прокручувати сторінку в М3 (lazy / infinite)
    api: Optional[ApiSpec] = None            # налаштування М2 (None = endpoint не знайдено)
    headers: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PageSpec":
        d = dict(d)
        api = d.pop("api", None)
        spec = PageSpec(**d)
        if api:
            spec.api = ApiSpec(**api)
        return spec


@dataclass
class ProductRecord:
    name: str
    price: Optional[float]
    price_raw: str = ""
    position: int = 0


@dataclass
class CollectResult:
    """Те, що повертає будь-який збирач. Вимірювання ресурсів робить раннер, не збирач."""
    records: list[ProductRecord] = field(default_factory=list)
    status_code: Optional[int] = None
    bytes_wire: int = 0          # байти, отримані з мережі (стиснені, як на дроті)
    bytes_body: int = 0          # байти тіл відповідей після розпакування
    n_requests: int = 0          # кількість HTTP-запитів
    n_xhr: int = 0               # кількість XHR/fetch-запитів (для кореляційного аналізу)
    html_size: int = 0           # розмір початкового HTML (для кореляційного аналізу)
    antibot_hits: int = 0        # M12: 403 / 429 / CAPTCHA-маркери
    error_type: Optional[str] = None   # timeout | network | http_403 | http_429 | captcha | parse | other
    error_msg: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ICollector(ABC):
    """Єдиний програмний інтерфейс для всіх методів збору.

    Раннер працює лише з цим контрактом і не знає деталей реалізації,
    тому всі методи вимірюються за однаковою процедурою.
    """
    name: str = "base"

    @abstractmethod
    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        ...

    def is_applicable(self, page: PageSpec) -> bool:
        """Чи можна застосувати метод до сторінки (М2 – лише якщо є endpoint)."""
        return True
