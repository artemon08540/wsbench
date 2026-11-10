"""Пошук і перевірка внутрішнього JSON-API сторінки для методу М2.

Для кожної сторінки з файлу вибірки:
  1. сторінка відкривається в headless-браузері (як у М3), і записуються всі
     JSON-відповіді на фонові запити (XHR/fetch) разом із самими запитами;
  2. у кожній відповіді шукається список об'єктів, у якому найбільше назв товарів,
     що збігаються з назвами на відрендереній сторінці; так визначаються
     items_path (шлях до списку), name_field і price_field;
  3. найкращий запит повторюється без браузера й без cookie – саме так його
     виконуватиме М2. Якщо відповідь та сама – endpoint придатний;
  4. перевіряється robots.txt для адреси API (це може бути інший домен).

Результат: results/find_api/api_found.yaml – готові блоки «api:» для config,
і find_api.json – подробиці для перевірки.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
import yaml

from ..collectors.common import DEFAULT_HEADERS, dig, extract_from_html, parse_price
from ..domain import ApiSpec, PageSpec
from .inspect_site import check_robots

# Заголовки запиту, які не переносимо в конфіг: їх ставить сам httpx/браузер,
# або це дані сесії (cookie, авторизація) – М2 працює без входу і без cookie.
SKIP_HEADERS = {"cookie", "authorization", "user-agent", "accept-encoding", "accept-language", "referer",
                "origin", "host", "connection", "content-length", "priority", "accept"}
SESSION_PARAM = re.compile(r"sess|token|uid|visitor|_ga|gclid", re.I)


@dataclass
class ApiResult:
    page_id: str
    url: str
    checked_at: str = ""
    dom_names: int = 0
    json_responses: int = 0
    status: str = "not_found"      # found | partial | replay_mismatch | not_found | replay_failed | robots_disallow
    api: dict[str, Any] | None = None
    matched_names: int = 0
    matched_prices: int = 0
    items_in_list: int = 0
    replay_status: int | None = None
    replay_items: int | None = None
    replay_match: int = 0
    robots: str = ""
    note: str = ""
    others: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(s).lower()).split())


def _key(s: str, n: int = 3) -> str:
    return " ".join(_norm(s).split()[:n])


def _flatten(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """{'price.current': 999, 'title': '...'} для одного елемента списку (до 3 рівнів вкладеності)."""
    out: dict[str, Any] = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}.", depth + 1))
    elif isinstance(obj, list):
        if obj and depth <= 2:
            out.update(_flatten(obj[0], f"{prefix}0.", depth + 1))
    else:
        out[prefix[:-1]] = obj
    return out


def _lists_of_dicts(obj: Any, path: str = "", depth: int = 0):
    """Усі списки об'єктів у JSON разом зі шляхом до них ('data.products.items')."""
    if depth > 8:
        return
    if isinstance(obj, list):
        if len(obj) >= 3 and all(isinstance(x, dict) for x in obj[:5]):
            yield path, obj
        for i, x in enumerate(obj[:3]):
            yield from _lists_of_dicts(x, f"{path}.{i}" if path else str(i), depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _lists_of_dicts(v, f"{path}.{k}" if path else k, depth + 1)


def _to_price(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and len(v) < 40:
        return parse_price(v)
    return None


def best_mapping(data: Any, dom_names: list[str], dom_prices: list[float]) -> dict | None:
    """Шукає в JSON список товарів і поля назви/ціни, що найкраще збігаються зі сторінкою."""
    dom_keys = {_key(n) for n in dom_names}
    dom_norm = [_norm(n) for n in dom_names]
    prices = {round(p, 2) for p in dom_prices if p}
    best = None
    for path, items in _lists_of_dicts(data):
        flat = [_flatten(it) for it in items]
        fields = {k for f in flat[:10] for k in f}
        # назва: рядкове поле з найбільшою кількістю збігів
        name_field, name_hits = None, 0
        for k in fields:
            hits = 0
            for f in flat:
                v = f.get(k)
                # назва – текст із пробілами; посилання й «slug» (samsung-tv-55) пропускаємо
                if not isinstance(v, str) or len(v) < 6 or " " not in v.strip() or v.startswith(("http", "/")):
                    continue
                nv = _norm(v)
                if _key(v) in dom_keys or any(nv and (nv in d or d in nv) for d in dom_norm):
                    hits += 1
            if hits > name_hits:
                name_field, name_hits = k, hits
        if not name_field or name_hits < 3:
            continue
        # ціна: числове поле, значення якого найчастіше дорівнюють цінам на сторінці
        price_field, price_hits = "", 0
        for k in fields:
            if k == name_field:
                continue
            hits = sum(1 for f in flat if (p := _to_price(f.get(k))) is not None and round(p, 2) in prices)
            if hits > price_hits:
                price_field, price_hits = k, hits
        score = (name_hits, price_hits, -len(path))
        if best is None or score > best["score"]:
            best = {"score": score, "items_path": path, "name_field": name_field, "price_field": price_field,
                    "name_hits": name_hits, "price_hits": price_hits, "n_items": len(items)}
    return best


def _strip_session(url: str) -> str:
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not SESSION_PARAM.search(k)]
    return urlunparse(p._replace(query=urlencode(q)))


def _count_matches(names: list[str], dom_names: list[str]) -> int:
    dom_keys = {_key(n) for n in dom_names}
    dom_norm = [_norm(n) for n in dom_names]
    return sum(1 for n in names if _key(n) in dom_keys or any(_norm(n) in d or d in _norm(n) for d in dom_norm))


def replay(page: PageSpec, api: ApiSpec, dom_names: list[str], timeout_s: int = 30) -> tuple[int | None, int | None, int, str]:
    """Повторює запит так само, як це зробить М2: без браузера. Повертає
    (HTTP-код, товарів у відповіді, скільки з них є на сторінці, помилка)."""
    headers = {**DEFAULT_HEADERS, "Accept": "application/json", "Referer": page.url, **page.headers, **api.headers}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
            if api.warmup:   # спершу сама сторінка – отримати cookie, які сайт дає кожному відвідувачу
                client.get(page.url, headers={"Accept": DEFAULT_HEADERS["Accept"]})
                time.sleep(1)
            r = client.request(api.method, api.url.replace("{uuid}", str(uuid.uuid4())),
                               params=api.params or None, json=api.json_body)
    except httpx.HTTPError as e:
        return None, None, 0, f"{type(e).__name__}"
    if r.status_code >= 400:
        return r.status_code, None, 0, f"HTTP {r.status_code}"
    try:
        items = dig(r.json(), api.items_path) or []
    except ValueError:
        return r.status_code, None, 0, "відповідь не JSON"
    names = [str(dig(it, api.name_field)) for it in items if dig(it, api.name_field)]
    return r.status_code, len(names), _count_matches(names, dom_names), ""


def find_api(page: PageSpec, headless: bool = True, log=print) -> ApiResult:
    from playwright.sync_api import TimeoutError as PwTimeout
    from playwright.sync_api import sync_playwright

    from ..collectors.headless import VIEWPORT, apply_page_headers, prepare_page

    res = ApiResult(page.page_id, page.url, datetime.now().isoformat(timespec="seconds"))
    captured: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], viewport=VIEWPORT)
        apply_page_headers(ctx, page)
        tab = ctx.new_page()

        def on_response(resp):
            try:
                req = resp.request
                if req.resource_type not in ("xhr", "fetch") or resp.status >= 400:
                    return
                body = resp.body()
                if len(body) < 500:
                    return
                data = json.loads(body)
                captured.append({"url": resp.url, "method": req.method, "post": req.post_data,
                                 "headers": req.headers, "data": data})
            except Exception:  # noqa: BLE001 – не JSON або відповідь недоступна
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
            prepare_page(tab, replace(page, scroll=True))
            recs = extract_from_html(tab.content(), page)
        finally:
            ctx.close()
            browser.close()

    res.dom_names, res.json_responses = len(recs), len(captured)
    if not recs:
        res.note = "на сторінці не знайдено карток (перевірте селектори)"
        return res

    scored = []
    for c in captured:
        m = best_mapping(c["data"], [r.name for r in recs], [r.price for r in recs if r.price])
        if m:
            scored.append((m["score"], m, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    res.others = [f"{c['method']} {c['url'][:140]}  назв: {m['name_hits']}" for _, m, c in scored[1:4]]
    if not scored:
        res.note = ("серед JSON-відповідей немає списку товарів – дані приходять у HTML "
                    "(сторінка SSR або фрагменти HTML), М2 для цієї сторінки не застосовний")
        return res

    _, m, c = scored[0]
    res.matched_names, res.matched_prices, res.items_in_list = m["name_hits"], m["price_hits"], m["n_items"]
    hdrs = {k: v for k, v in c["headers"].items() if k.lower() not in SKIP_HEADERS and not k.startswith(":")
            and not k.lower().startswith("sec-")}
    body = None
    if c["post"]:
        try:
            body = json.loads(c["post"])
        except ValueError:
            res.note = "тіло POST-запиту не JSON – М2 його не відтворить"
    if c["method"] == "POST":
        hdrs.pop("content-type", None)    # httpx поставить сам для json=
    api = ApiSpec(url=_strip_session(c["url"]), method=c["method"], json_body=body, headers=hdrs,
                  items_path=m["items_path"], name_field=m["name_field"], price_field=m["price_field"])

    dom = [r.name for r in recs]
    time.sleep(2)
    res.replay_status, res.replay_items, res.replay_match, err = replay(page, api, dom)
    if res.replay_items is None and api.url != c["url"]:      # без «сесійних» параметрів не працює
        api = replace(api, url=c["url"])
        time.sleep(2)
        res.replay_status, res.replay_items, res.replay_match, err = replay(page, api, dom)
    if res.replay_items and res.replay_match < 0.8 * res.dom_names:
        # без браузера API віддає інші товари: пробуємо, як справжній клієнт, спершу відкрити сторінку (cookie)
        time.sleep(2)
        w = replay(page, replace(api, warmup=True), dom)
        if w[1] and w[2] > res.replay_match:
            api = replace(api, warmup=True)
            res.replay_status, res.replay_items, res.replay_match, err = w
    res.robots = check_robots(api.url)
    res.api = {k: v for k, v in asdict(api).items() if v not in (None, {}, "", False) or k == "price_field"}

    if res.robots == "no":
        res.status = "robots_disallow"
        res.note = "robots.txt забороняє цю адресу – М2 для сторінки не використовуємо"
    elif not res.replay_items:
        res.status = "replay_failed"
        res.note = f"у браузері endpoint працює, а без браузера – ні ({err or 'порожньо'}): потрібні cookie чи захист"
    elif res.replay_items < 0.8 * res.dom_names:
        # напр. віджет відгуків чи реклами з кількома товарами – це не список каталогу
        res.status = "partial"
        res.note = (f"JSON містить лише частину товарів сторінки ({res.replay_items} з {res.dom_names}) – "
                    "ймовірно віджет відгуків/реклами, а не список каталогу; М2 не застосовний")
    elif res.replay_match < 0.8 * res.dom_names:
        res.status = "replay_mismatch"
        res.note = (f"без браузера API віддає інший набір товарів: зі сторінкою збігається {res.replay_match} з "
                    f"{res.dom_names} (залежить від стану сесії в браузері)")
    else:
        res.status = "found"
        if api.warmup:
            res.note = "потрібен попередній запит сторінки (cookie відвідувача) – warmup: true"
        if not api.price_field:
            res.note = (res.note + "; " if res.note else "") + "ціну в JSON не знайдено автоматично – перевірте вручну"
    return res


def save(results: list[ApiResult], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "find_api.json"
    js.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    snippet = {r.page_id: {"api": r.api if r.status == "found" else None,
                           "m2_status": r.status, "note": r.note} for r in results}
    ym = out_dir / "api_found.yaml"
    ym.write_text(yaml.safe_dump(snippet, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    return ym, js
