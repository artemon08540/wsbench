"""M14 – трудомісткість підготовки збирача (п. 6.4 ТЗ).

(а) хвилини роботи розробника – ручний журнал journal/effort_log.csv (page_id, method, minutes, ...);
(б) обсяг конфігурації адаптера (LOC) – рахується автоматично з config/sites.yaml як кількість
    непорожніх параметрів, які людина мусила задати для методу на сторінці:
      М1 – селектори картки/назви/ціни + власні заголовки;
      М2 – усі непорожні поля блоку api (адреса, параметри, заголовки, шляхи до полів);
      М3 – селектори + селектор очікування + прапорець прокручування + заголовки;
      М4 – об'єднання налаштувань М1 і М3 + поріг перемикання (1).
Застереження (обов'язково в роботі): (а) суб'єктивна оцінка однієї людини з ефектом навчання.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def _count_leaves(x) -> int:
    if isinstance(x, dict):
        return sum(_count_leaves(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_count_leaves(v) for v in x)
    return 0 if x in (None, "", False) else 1


def config_loc(sample: Path) -> pd.DataFrame:
    data = yaml.safe_load(Path(sample).read_text(encoding="utf-8"))
    rows = []
    for p in (data["pages"] if isinstance(data, dict) else data):
        sel = {k: p.get(k) for k in ("card_selector", "name_selector", "price_selector")}
        m1 = set(k for k, v in sel.items() if v)
        hdr = _count_leaves(p.get("headers") or {})
        m3 = m1 | ({"wait_selector"} if p.get("wait_selector") else set()) | ({"scroll"} if p.get("scroll") else set())
        api = p.get("api")
        rows += [
            {"page_id": p["page_id"], "method": "M1", "loc": len(m1) + hdr},
            {"page_id": p["page_id"], "method": "M2", "loc": (_count_leaves(api) + hdr) if api else None},
            {"page_id": p["page_id"], "method": "M3", "loc": len(m3) + hdr},
            {"page_id": p["page_id"], "method": "M4", "loc": len(m1 | m3) + hdr + 1},
        ]
    return pd.DataFrame(rows)


def effort_table(sample: Path, log: Path, page_ids: list[str] | None = None) -> pd.DataFrame:
    loc = config_loc(sample)
    if page_ids is not None:
        loc = loc[loc["page_id"].isin(page_ids)]
    t = loc.groupby("method")["loc"].agg(["median", "mean", "count"]).rename(
        columns={"median": "LOC (медіана)", "mean": "LOC (середнє)", "count": "сторінок"})
    if Path(log).exists():
        lg = pd.read_csv(log)
        if not lg.empty and "minutes" in lg:
            lg["minutes"] = pd.to_numeric(lg["minutes"], errors="coerce")
            if page_ids is not None:
                lg = lg[lg["page_id"].isin(page_ids)]
            mins = lg.groupby("method")["minutes"].agg(["median", "count"]).rename(
                columns={"median": "хвилини (медіана)", "count": "записів у журналі"})
            t = t.join(mins, how="left")
    return t
