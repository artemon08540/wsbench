"""Швидкий відбір кандидатів у вибірку без налаштування селекторів.

Для кожної адреси зі списку:
  1. robots.txt – чи дозволено обходити цей шлях;
  2. сирий HTML через httpx (так його бачить М1): код відповіді, ознаки блокування;
  3. сторінка в headless-браузері (так її бачить М3) з поступовим прокручуванням;
     картки товарів знаходяться евристично: найчисленніша група однотипних
     найменших елементів, що містять посилання, зображення і ціну;
  4. частка назв товарів зі сторінки, присутніх у сирому HTML,
     звідки тип: < 20 % – spa, 20–80 % – hybrid, ≥ 80 % – ssr.

Евристика дає попередню оцінку. Остаточний тип визначає check-site після того,
як для сторінки задано селектори в config/sites.yaml.
"""
from __future__ import annotations

import csv
import html as htmllib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..collectors.common import DEFAULT_HEADERS, detect_antibot
from .inspect_site import check_robots, classify

PROBE_JS = r"""
() => {
  const priceRe = /(\d[\d\s .,]*\s?(₴|грн|zł|zl|€|£|\$|kr|Kč|PLN|EUR|UAH|,-))|((₴|€|£|\$)\s?\d)/i;
  const all = [...document.querySelectorAll('body *')];
  const cand = all.filter(e => {
    const t = (e.innerText || '').trim();
    return t.length > 15 && t.length < 1200 && priceRe.test(t)
      && (e.matches('a[href]') || e.querySelector('a[href]')) && e.querySelector('img,picture');
  });
  const set = new Set(cand);
  const minimal = cand.filter(e => ![...e.querySelectorAll('*')].some(c => set.has(c)));
  const sig = e => e.tagName.toLowerCase() + (e.classList[0] ? '.' + e.classList[0] : '');
  const groups = {};
  minimal.forEach(e => { (groups[sig(e)] = groups[sig(e)] || []).push(e); });
  const best = Object.entries(groups).sort((a, b) => b[1].length - a[1].length)[0];
  if (!best) return {cards: 0, selector: '', names: [], name_selector: '', price_selector: ''};
  const cards = best[1];
  const bad = /opini|ocen|review|arvostel|dostaw|delivery|koszyk|cart|відгук|отзыв|доставка|кошик/i;
  const nameEl = c => {
    const els = [...c.querySelectorAll('*')].filter(x => {
      const t = (x.innerText || '').trim();
      return t.length >= 12 && t.length < 220 && !priceRe.test(t) && !bad.test(t)
        && [...x.children].every(ch => (ch.innerText || '').trim() !== t);
    });
    els.sort((a, b) => b.innerText.length - a.innerText.length);
    return els[0] || null;
  };
  const priceEl = c => {
    const els = [...c.querySelectorAll('*')].filter(x => priceRe.test((x.innerText || '').trim())
      && (x.innerText || '').trim().length < 40);
    els.sort((a, b) => a.innerText.length - b.innerText.length);
    return els[0] || null;
  };
  const most = arr => { const m = {}; arr.filter(Boolean).forEach(s => m[s] = (m[s] || 0) + 1);
    return (Object.entries(m).sort((a, b) => b[1] - a[1])[0] || [''])[0]; };
  const names = [], nsel = [], psel = [];
  cards.forEach(c => {
    const n = nameEl(c); if (n) { names.push(n.innerText.trim().replace(/\s+/g, ' ')); nsel.push(sig(n)); }
    const p = priceEl(c); if (p) psel.push(sig(p));
  });
  return {cards: cards.length, selector: best[0], names, name_selector: most(nsel), price_selector: most(psel)};
}
"""


@dataclass
class Candidate:
    cand_id: str
    url: str
    checked_at: str = ""
    robots: str = "?"
    raw_status: int | None = None
    antibot: str = ""
    final_url: str = ""
    title: str = ""
    rendered_cards: int = 0
    names_found: int = 0
    names_in_raw: int = 0
    raw_share: float | None = None
    render_type_auto: str = "unknown"
    card_selector: str = ""
    name_selector: str = ""
    price_selector: str = ""
    sample: list[str] = field(default_factory=list)
    error: str = ""


def _norm(s: str) -> str:
    s = htmllib.unescape(s)
    s = s.replace("\\u002F", "/").replace('\\"', '"').replace("\\/", "/")
    return re.sub(r"\s+", " ", s).lower()


def _key(name: str) -> str:
    return " ".join(name.split(" ")[:4]).lower()


def load_candidates(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        out.append((parts[0], parts[1]) if len(parts) > 1 else (urlparse(parts[0]).netloc, parts[0]))
    return out


def discover(cands: list[tuple[str, str]], headless: bool = True, log=print) -> list[Candidate]:
    from playwright.sync_api import sync_playwright

    from ..collectors.headless import VIEWPORT

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        for i, (cid, url) in enumerate(cands, 1):
            c = Candidate(cid, url, datetime.now().isoformat(timespec="seconds"))
            log(f"[{i}/{len(cands)}] {cid}: {url}")
            c.robots = check_robots(url)
            raw = ""
            try:
                r = httpx.get(url, headers=DEFAULT_HEADERS, timeout=30, follow_redirects=True)
                c.raw_status, raw = r.status_code, r.text
                _, err = detect_antibot(r.status_code, r.text)
                c.antibot = err or ""
            except httpx.HTTPError as e:
                c.antibot = f"raw_error: {type(e).__name__}"
            time.sleep(2)
            ctx = browser.new_context(user_agent=DEFAULT_HEADERS["User-Agent"], viewport=VIEWPORT)
            tab = ctx.new_page()
            try:
                tab.goto(url, wait_until="domcontentloaded", timeout=60_000)
                tab.wait_for_timeout(3500)
                for _ in range(12):
                    tab.mouse.wheel(0, 700)
                    tab.wait_for_timeout(400)
                tab.wait_for_timeout(1500)
                c.final_url, c.title = tab.url, (tab.title() or "")[:80]
                res = tab.evaluate(PROBE_JS)
                names = [n for n in res["names"] if n]
                c.rendered_cards, c.names_found = res["cards"], len(names)
                c.card_selector, c.name_selector, c.price_selector = res["selector"], res["name_selector"], res["price_selector"]
                c.sample = names[:3]
                rn = _norm(raw)
                c.names_in_raw = sum(1 for n in names if _key(n) in rn)
            except Exception as e:  # noqa: BLE001
                c.error = f"{type(e).__name__}: {str(e)[:150]}"
            finally:
                ctx.close()
            blocked = c.raw_status is None or c.raw_status >= 400 or bool(c.antibot)
            c.raw_share, c.render_type_auto = classify(c.names_in_raw, c.names_found, blocked)
            if c.raw_share is not None:
                c.raw_share = round(c.raw_share, 2)
            log(f"      -> {c.render_type_auto:8} карток {c.rendered_cards:3}, у сирому HTML {c.names_in_raw}/{c.names_found}, "
                f"HTTP {c.raw_status}, robots {c.robots}{'  ' + c.antibot if c.antibot else ''}{'  ' + c.error if c.error else ''}")
            results.append(c)
        browser.close()
    return results


def save(results: list[Candidate], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "discover.csv"
    rows = [asdict(c) | {"sample": " | ".join(c.sample)} for c in results]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    return path
