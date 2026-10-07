"""М3 – headless-браузер (Chromium через Playwright) із повним рендерингом.

Політика очікування (фіксується в розділі 2 роботи дослівно):
  1) перехід на сторінку до події DOMContentLoaded;
  2) очікування появи селектора товару (wait_selector або card_selector), тайм-аут 15 с;
  3) додатково – очікування networkidle, не довше 5 с;
  4) якщо scroll=True: ПОСТУПОВЕ прокручування – кроками по 80 % висоти вікна з паузою 0,4 с
     (щоб спрацювали ліниве малювання карток і довантаження, які реагують лише на видиму
     частину сторінки); після досягнення низу – пауза 1 с; зупинка, коли кількість ЗАПОВНЕНИХ
     карток (з непорожньою назвою) не зросла 2 рази поспіль, але не більше 20 досягнень низу
     і не довше 30 с.
  Вікно браузера: 1366×900 (фіксується як умова експерименту).
Дані вилучаються з готового DOM (page.content()) тією самою функцією, що й у М1.

Ізоляція: кожен прогін запускає НОВИЙ процес браузера і новий контекст (порожній кеш і
cookie). Час запуску браузера зберігається окремо (extra.browser_launch_ms), тож у
розділі 3 його можна показати і з урахуванням, і без.
"""
from __future__ import annotations

import time

from ..domain import CollectResult, ICollector, PageSpec
from .common import DEFAULT_HEADERS, detect_antibot, extract_from_html

SELECTOR_TIMEOUT_MS = 15_000
NETWORKIDLE_TIMEOUT_MS = 5_000
SCROLL_PAUSE_MS = 1_000       # пауза після досягнення низу сторінки
SCROLL_STEP_PAUSE_MS = 400    # пауза після кожного кроку прокручування
SCROLL_STEP_FRACTION = 0.8    # крок = 80 % висоти вікна
SCROLL_MAX = 20               # максимум досягнень низу
SCROLL_PATIENCE = 2           # стоп, якщо заповнених карток не побільшало 2 рази поспіль
SCROLL_BUDGET_S = 30          # загальне обмеження часу прокручування
VIEWPORT = {"width": 1366, "height": 900}


def apply_page_headers(ctx, page: PageSpec) -> None:
    """Заголовки зі sites.yaml для браузера: Cookie – як справжні cookie домену, решта – як HTTP-заголовки.
    Так М1 і М3 надсилають сайту однакові налаштування (напр. вибір країни intl=nosplash)."""
    from urllib.parse import urlparse
    headers = dict(page.headers or {})
    cookie = headers.pop("Cookie", None) or headers.pop("cookie", None)
    if cookie:
        p = urlparse(page.url)
        origin = f"{p.scheme}://{p.netloc}/"
        cookies = []
        for part in cookie.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                cookies.append({"name": k, "value": v, "url": origin})
        ctx.add_cookies(cookies)
    if headers:
        ctx.set_extra_http_headers(headers)


class HeadlessCollector(ICollector):
    name = "M3"

    def __init__(self, headless: bool = True):
        self.headless = headless

    def collect(self, page: PageSpec, timeout_s: int) -> CollectResult:
        from playwright.sync_api import Error as PwError
        from playwright.sync_api import TimeoutError as PwTimeout
        from playwright.sync_api import sync_playwright

        res = CollectResult()
        net = {"wire": 0, "requests": 0, "xhr": 0}
        main_status: dict[str, int] = {}

        try:
            with sync_playwright() as pw:
                t0 = time.perf_counter()
                browser = pw.chromium.launch(headless=self.headless)
                res.extra["browser_launch_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                try:
                    ctx = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], locale="uk-UA",
                                              viewport=VIEWPORT)
                    apply_page_headers(ctx, page)
                    ctx.set_default_timeout(timeout_s * 1000)
                    tab = ctx.new_page()

                    # Трафік вимірюємо через Chrome DevTools Protocol: encodedDataLength –
                    # реальна кількість байтів, отриманих із мережі (зі стисненням і заголовками).
                    cdp = ctx.new_cdp_session(tab)
                    cdp.send("Network.enable")

                    def on_sent(ev):
                        net["requests"] += 1
                        if ev.get("type") in ("XHR", "Fetch"):
                            net["xhr"] += 1

                    def on_finished(ev):
                        net["wire"] += int(ev.get("encodedDataLength", 0))

                    def on_resp(ev):
                        if ev.get("type") == "Document" and "status" not in main_status:
                            main_status["status"] = ev["response"]["status"]

                    cdp.on("Network.requestWillBeSent", on_sent)
                    cdp.on("Network.loadingFinished", on_finished)
                    cdp.on("Network.responseReceived", on_resp)

                    resp = tab.goto(page.url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
                    res.status_code = resp.status if resp else main_status.get("status")

                    sel = page.wait_selector or page.card_selector
                    try:
                        tab.wait_for_selector(sel, timeout=SELECTOR_TIMEOUT_MS)
                    except PwTimeout:
                        res.extra["selector_timeout"] = True
                    try:
                        tab.wait_for_load_state("networkidle", timeout=NETWORKIDLE_TIMEOUT_MS)
                    except PwTimeout:
                        res.extra["networkidle_timeout"] = True

                    if page.scroll:
                        res.extra["scrolls"] = self._scroll(tab, page.card_selector, page.name_selector)

                    html = tab.content()
                    res.bytes_body = len(html.encode("utf-8"))
                    res.antibot_hits, err = detect_antibot(res.status_code, html)
                    if err:
                        res.error_type = err
                    elif res.status_code is not None and res.status_code >= 400:
                        res.error_type, res.error_msg = "other", f"HTTP {res.status_code}"
                    else:
                        try:
                            res.records = extract_from_html(html, page)
                        except Exception as e:  # noqa: BLE001
                            res.error_type, res.error_msg = "parse", repr(e)
                    ctx.close()
                finally:
                    browser.close()
        except PwTimeout as e:
            res.error_type, res.error_msg = "timeout", str(e)[:300]
        except PwError as e:
            msg = str(e)
            res.error_type = "network" if "net::" in msg else "other"
            res.error_msg = msg[:300]

        res.bytes_wire = net["wire"]
        res.n_requests = net["requests"]
        res.n_xhr = net["xhr"]
        return res

    @staticmethod
    def _scroll(tab, card_selector: str, name_selector: str | None = None) -> int:
        """Поступове прокручування. Повертає кількість кроків."""
        js_filled = """([card, name]) => [...document.querySelectorAll(card)].filter(c => {
            if (!name) return true; const n = c.querySelector(name); return n && n.textContent.trim().length > 0; }).length"""
        filled = lambda: tab.evaluate(js_filled, [card_selector, name_selector])  # noqa: E731
        t0 = time.monotonic()
        best, stale, bottoms, steps = filled(), 0, 0, 0
        while bottoms < SCROLL_MAX and stale < SCROLL_PATIENCE and time.monotonic() - t0 < SCROLL_BUDGET_S:
            at_bottom = tab.evaluate(
                f"""() => {{ window.scrollBy(0, Math.round(window.innerHeight * {SCROLL_STEP_FRACTION}));
                    return window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 5; }}""")
            steps += 1
            tab.wait_for_timeout(SCROLL_STEP_PAUSE_MS)
            if not at_bottom:
                continue
            bottoms += 1
            tab.wait_for_timeout(SCROLL_PAUSE_MS)
            now = filled()
            stale = stale + 1 if now <= best else 0
            best = max(best, now)
        return steps
