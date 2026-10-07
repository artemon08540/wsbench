"""Попередня перевірка сторінки перед включенням у вибірку (п. 7.1 і розділ 13 ТЗ):

  * robots.txt – чи дозволено обходити цей шлях;
  * тип рендерингу – кількість карток у сирому HTML (як бачить М1) проти кількості
    в DOM після рендерингу (як бачить М3):  <20 % -> spa,  20–80 % -> hybrid,  >=80 % -> ssr;
  * кандидати для М2 – JSON-відповіді, які сторінка сама завантажує (XHR/fetch),
    з позначкою, чи містять вони назву першого товару зі сторінки;
  * вбудовані дані (__NEXT_DATA__, __NUXT__ тощо) – теж потенційне джерело для М2.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ..collectors.common import DEFAULT_HEADERS, count_cards, detect_antibot, extract_from_html
from ..domain import PageSpec

EMBEDDED_MARKERS = ("__NEXT_DATA__", "__NUXT__", "__INITIAL_STATE__", "__APOLLO_STATE__", "window.__data", "application/ld+json")


@dataclass
class SiteCheck:
    page_id: str
    url: str
    checked_at: str
    robots_allowed: str = "?"           # yes | no | no_robots | error
    raw_status: int | None = None
    raw_cards: int = 0
    raw_shells: int = 0                 # «коробки» карток у сирому HTML, включно з порожніми
    rendered_cards: int = 0
    raw_share: float | None = None      # частка карток, присутніх у сирому HTML
    render_type_auto: str = "unknown"
    antibot: str = ""
    embedded_data: list[str] = field(default_factory=list)
    json_candidates: list[dict] = field(default_factory=list)
    first_names: list[str] = field(default_factory=list)
    final_url: str = ""


def check_robots(url: str, ua: str = "*") -> str:
    p = urlparse(url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        r = httpx.get(robots_url, headers=DEFAULT_HEADERS, timeout=15, follow_redirects=True)
    except httpx.HTTPError:
        return "error"
    if r.status_code == 404:
        return "no_robots"
    if r.status_code in (401, 403, 429):
        return f"blocked_{r.status_code} (перевірте вручну в браузері)"
    if r.status_code >= 400:
        return f"error_{r.status_code}"
    rp = RobotFileParser()
    rp.parse(r.text.splitlines())
    return "yes" if rp.can_fetch(ua, url) else "no"


def classify(raw: int, rendered: int, raw_blocked: bool = False) -> tuple[float | None, str]:
    if raw_blocked:
        return None, "blocked"       # сирий HTML не отримано (403/429/CAPTCHA) – тип визначити не можна
    if rendered == 0:
        return None, "unknown"
    share = raw / rendered
    if share < 0.2:
        return share, "spa"
    if share < 0.8:
        return share, "hybrid"
    return share, "ssr"


def inspect_page(page: PageSpec, headless: bool = True, scroll: bool | None = None,
                 debug_dir: Path | None = None) -> SiteCheck:
    """debug_dir: куди зберегти raw.html, rendered.html і screenshot.png (для перевірки «очима»)."""
    from playwright.sync_api import TimeoutError as PwTimeout
    from playwright.sync_api import sync_playwright

    chk = SiteCheck(page.page_id, page.url, datetime.now().isoformat(timespec="seconds"))
    chk.robots_allowed = check_robots(page.url)

    # 1) Сирий HTML (як М1)
    try:
        r = httpx.get(page.url, headers={**DEFAULT_HEADERS, **page.headers}, timeout=30, follow_redirects=True)
        chk.raw_status = r.status_code
        chk.raw_shells = count_cards(r.text, page.card_selector)
        chk.raw_cards = len(extract_from_html(r.text, page))   # лише картки з назвою
        hits, err = detect_antibot(r.status_code, r.text)
        if err:
            chk.antibot = err
        chk.embedded_data = [m for m in EMBEDDED_MARKERS if m in r.text]
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "raw.html").write_text(r.text, encoding="utf-8")
    except httpx.HTTPError as e:
        chk.antibot = f"raw_error: {e!r}"[:200]

    time.sleep(2)  # ввічливість: пауза між зверненнями до домену

    # 2) Відрендерений DOM (як М3) + перехоплення JSON-відповідей
    json_resps: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        from ..collectors.headless import VIEWPORT
        ctx = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], locale="uk-UA", viewport=VIEWPORT)
        from ..collectors.headless import apply_page_headers
        apply_page_headers(ctx, page)
        tab = ctx.new_page()

        def on_response(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if resp.request.resource_type in ("xhr", "fetch") and "json" in ct:
                    body = resp.body()
                    json_resps.append({"url": resp.url, "method": resp.request.method,
                                       "status": resp.status, "bytes": len(body), "_body": body[:2_000_000]})
            except Exception:  # noqa: BLE001
                pass

        tab.on("response", on_response)
        try:
            tab.goto(page.url, wait_until="domcontentloaded", timeout=60_000)
            try:
                tab.wait_for_selector(page.wait_selector or page.card_selector, timeout=15_000)
            except PwTimeout:
                pass
            try:
                tab.wait_for_load_state("networkidle", timeout=5_000)
            except PwTimeout:
                pass
            if scroll if scroll is not None else page.scroll:
                from ..collectors.headless import HeadlessCollector
                HeadlessCollector._scroll(tab, page.card_selector, page.name_selector)
            html = tab.content()
            if debug_dir:
                (debug_dir / "rendered.html").write_text(html, encoding="utf-8")
                tab.screenshot(path=str(debug_dir / "screenshot.png"))
            chk.final_url = tab.url
            recs = extract_from_html(html, page)
            chk.rendered_cards = len(recs)   # лише картки з назвою
            chk.first_names = [r.name for r in recs[:3]]
        finally:
            ctx.close()
            browser.close()

    raw_blocked = chk.raw_status is None or chk.raw_status >= 400 or bool(chk.antibot)
    chk.raw_share, chk.render_type_auto = classify(chk.raw_cards, chk.rendered_cards, raw_blocked)

    probe = chk.first_names[0].lower()[:25] if chk.first_names else None
    seen, uniq = set(), []
    for jr in sorted(json_resps, key=lambda x: -x["bytes"]):
        if jr["url"] not in seen:
            seen.add(jr["url"])
            uniq.append(jr)
    for jr in uniq[:15]:
        body = jr.pop("_body")
        try:
            text = json.dumps(json.loads(body), ensure_ascii=False).lower()
        except Exception:  # noqa: BLE001
            text = body.decode("utf-8", "ignore").lower()
        jr["contains_first_product"] = bool(probe and probe in text)
        jr["url"] = urljoin(page.url, jr["url"])
        chk.json_candidates.append(jr)
    return chk
