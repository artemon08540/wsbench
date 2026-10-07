"""Блок дослідження стійкості (п. 7.4 ТЗ, гіпотеза 3, показник M13).

Ідея: збирач налаштовується на НАЙСТАРІШУ архівну версію сторінки (Wayback Machine) і без жодних
змін запускається на трьох версіях: ~24 міс. тому, ~12 міс. тому і поточній.
M13 (Selector Survival Rate) = частка пар (сторінка, новіша версія), на яких збирач досі працює
(Recall ≥ 90 % за еталоном цієї версії; якщо еталону немає – знайдено ≥ 90 % від очікуваної кількості).

Обмеження (обов'язково в «загрозах валідності»): архів не завжди відтворює JavaScript, тому для М3/М4
результати менш надійні; внутрішні API архівуються рідко, тож провал М2 на архіві може означати
відсутність копії, а не зміну API. Сайти, що відтворюються некоректно, виключаються з блоку з документуванням.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import yaml

from ..collectors.common import DEFAULT_HEADERS
from ..domain import ApiSpec, PageSpec
from ..groundtruth import load_ground_truth
from .runner import RunExecutor, Task, load_sample

WAYBACK = "https://web.archive.org/web/{ts}/{url}"
AVAILABILITY = "https://archive.org/wayback/available"
SURVIVAL_RECALL = 90.0


def archived_url(url: str, ts: str) -> str:
    return url if ts == "live" else WAYBACK.format(ts=ts, url=url)


def find_snapshot(url: str, target: date) -> Optional[str]:
    """Найближча до дати архівна копія (Wayback Availability API). Повертає позначку часу YYYYMMDDhhmmss."""
    try:
        r = httpx.get(AVAILABILITY, params={"url": url, "timestamp": target.strftime("%Y%m%d")},
                      headers=DEFAULT_HEADERS, timeout=30, follow_redirects=True)
        snap = r.json().get("archived_snapshots", {}).get("closest")
        return snap["timestamp"] if snap and snap.get("available") else None
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def suggest_snapshots(pages: list[PageSpec], months: tuple[int, ...] = (24, 12)) -> list[dict]:
    out = []
    today = date.today()
    for p in pages:
        snaps = []
        for m in months:
            snaps.append(find_snapshot(p.url, today - timedelta(days=round(m * 30.44))))
        out.append({"page_id": p.page_id, "url": p.url, "snapshots": [s or "НЕ ЗНАЙДЕНО" for s in snaps] + ["live"]})
    return out


@dataclass
class StabilityPage:
    base: PageSpec
    snapshots: list[str]                     # від найстарішої до "live"
    expected_counts: dict[str, int]


def load_stability(path: Path, sample: Path) -> list[StabilityPage]:
    """config/stability.yaml: для кожної сторінки – список версій і (необов'язково) селектори,
    налаштовані на НАЙСТАРІШУ версію (перекривають значення з sites.yaml)."""
    base = {p.page_id: p for p in load_sample(sample)}
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in cfg["pages"]:
        p = copy.deepcopy(base[e["page_id"]])
        for k in ("card_selector", "name_selector", "price_selector", "wait_selector", "scroll"):
            if k in e:
                setattr(p, k, e[k])
        if "api" in e:
            p.api = ApiSpec(**e["api"]) if e["api"] else None
        out.append(StabilityPage(p, [str(s) for s in e["snapshots"]], {str(k): v for k, v in (e.get("expected_counts") or {}).items()}))
    return out


def version_page(sp: StabilityPage, ts: str) -> PageSpec:
    p = copy.deepcopy(sp.base)
    p.page_id = f"{sp.base.page_id}@{ts}"
    p.url = archived_url(sp.base.url, ts)
    if p.api is not None and ts != "live":
        p.api.url = archived_url(p.api.url, ts)
    p.expected_count = sp.expected_counts.get(ts, p.expected_count)
    p.scroll = p.scroll and ts == "live"     # на архіві довантаження зазвичай не працює
    return p


def run_stability(stab: list[StabilityPage], collectors: dict, out_dir: Path, gt_dir: Path,
                  min_delay_s: float = 4.0, on_row=None) -> Path:
    tasks = []
    for sp in stab:
        for ts in sp.snapshots:
            vp = version_page(sp, ts)
            for m in collectors:
                tasks.append(Task(vp, m, 1))
    ex = RunExecutor(collectors, out_dir, block="stability", gt_dir=gt_dir, min_delay_s=min_delay_s, on_row=on_row)
    return ex.run(tasks)


def survival_table(runs_csv: Path, stab: list[StabilityPage], gt_dir: Path) -> pd.DataFrame:
    """Рядок на (сторінка, версія, метод): чи працює незмінний збирач на цій версії."""
    df = pd.read_csv(runs_csv)
    rows = []
    for sp in stab:
        oldest = sp.snapshots[0]
        for ts in sp.snapshots:
            pid = f"{sp.base.page_id}@{ts}"
            sub = df[df["page_id"] == pid]
            has_gt = load_ground_truth(gt_dir, pid) is not None
            for _, r in sub.iterrows():
                exp = sp.expected_counts.get(ts)
                if has_gt and pd.notna(r.get("recall")):
                    score, basis = float(r["recall"]), "еталон"
                elif exp:
                    score, basis = 100.0 * float(r.get("n_found") or 0) / exp, "к-сть/очікувана"
                else:
                    score, basis = float("nan"), "немає еталону й очікуваної к-сті"
                rows.append({"page_id": sp.base.page_id, "version": ts, "is_reference": ts == oldest,
                             "method": r["method"], "n_found": r.get("n_found"), "score_%": round(score, 1),
                             "basis": basis, "error_type": r.get("error_type", ""),
                             "works": (score >= SURVIVAL_RECALL) if score == score else None})
    return pd.DataFrame(rows)


def ssr_by_method(surv: pd.DataFrame) -> pd.DataFrame:
    """M13 для кожного методу: частка (сторінка, новіша версія) з works=True. Еталонна (найстаріша) версія не входить."""
    d = surv[(~surv["is_reference"]) & surv["works"].notna()]
    g = d.groupby("method")["works"]
    return pd.DataFrame({"N пар": g.size(), "Працює": g.sum().astype(int), "SSR, %": (100 * g.mean()).round(1)})
