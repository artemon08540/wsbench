"""Зведення результатів прогонів у таблицю (медіани за повторами) і службові експорти."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from ..domain import PageSpec

KEY_COLS = ["n_found", "recall", "t_ms", "traffic_mb", "ram_peak_mb", "cpu_s"]


def load_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in KEY_COLS + ["precision", "f1", "field_accuracy", "throughput"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


NA = "not_applicable"


def _mask_na(df: pd.DataFrame) -> pd.DataFrame:
    """Для незастосовних прогонів усі метрики = NaN (а не 0), щоб вони не потрапляли в медіани."""
    d = df.copy()
    if "error_type" in d:
        na = d["error_type"] == NA
        for c in KEY_COLS:
            if c in d:
                d.loc[na, c] = float("nan")
    return d


def pilot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сторінка × метод: медіани за повторами + CV часу (для обґрунтування кількості повторів)."""
    df = _mask_na(df)
    cols = [c for c in KEY_COLS if c in df and df[c].notna().any()]
    g = df.groupby(["page_id", "render_type", "method"])
    med = g[cols].median().round(2)
    med["t_cv_%"] = (g["t_ms"].std() / g["t_ms"].mean() * 100).round(1)
    med["errors"] = g["error_type"].apply(lambda s: ",".join(sorted({str(x) for x in s if isinstance(x, str) and x})))
    med["n_runs"] = g.size()
    return med.reset_index()


def _coverage(df: pd.DataFrame, by: list[str]) -> pd.Series:
    """Охоплення: на скількох сторінках метод застосовний, у вигляді «k/n»."""
    app = df[df["error_type"] != NA] if "error_type" in df else df
    n = df.groupby(by)["page_id"].nunique()
    k = app.groupby(by)["page_id"].nunique().reindex(n.index).fillna(0).astype(int)
    return (k.astype(str) + "/" + n.astype(str)).rename("охоплення")


def method_table(df: pd.DataFrame, by: tuple[str, ...] = ("method",)) -> pd.DataFrame:
    """Медіана медіан: спочатку за повторами на сторінці, потім за сторінками.
    Незастосовні прогони (not_applicable) виключено з усіх метрик; охоплення вказано окремо."""
    by = list(by)
    d = _mask_na(df)
    cols = [c for c in KEY_COLS if c in d and d[c].notna().any()]
    page = d.groupby(by + ["page_id"])[cols].median()
    out = page.groupby(level=by).median().round(2)
    out["охоплення"] = _coverage(df, by)
    return out


def render_table(df: pd.DataFrame) -> pd.DataFrame:
    """Те саме з розбивкою за типом рендерингу (ssr / spa / hybrid)."""
    return method_table(df, by=("render_type", "method"))


def export_sample_xlsx(pages: list[PageSpec], checks: dict[str, dict], out: Path) -> None:
    rows = []
    for i, p in enumerate(pages, start=1):
        c = checks.get(p.page_id, {})
        rows.append({
            "№": i, "page_id": p.page_id, "домен": p.site, "URL": p.url,
            "тип рендерингу": p.render_type,
            "тип рендерингу (авто)": c.get("render_type_auto", ""),
            "частка карток у сирому HTML": c.get("raw_share", ""),
            "тип пагінації": p.pagination,
            "очікувана кількість товарів": p.expected_count or c.get("rendered_cards", ""),
            "robots.txt": c.get("robots_allowed", ""),
            "дата перевірки robots.txt": c.get("checked_at", ""),
            "endpoint М2": "так" if p.api else "ні",
            "примітки": p.notes,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(out, index=False, sheet_name="sample")


def environment_info() -> dict:
    import psutil
    libs = {}
    for name in ["httpx", "beautifulsoup4", "lxml", "playwright", "psutil", "pandas", "scipy", "pyyaml", "openpyxl", "matplotlib"]:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    chromium = None
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            chromium = b.version
            b.close()
    except Exception as e:  # noqa: BLE001
        chromium = f"n/a ({e.__class__.__name__})"
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_logical": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": sys.version.split()[0],
        "libraries": libs,
        "chromium": chromium,
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
