"""Довідник показників M1–M14 (п. 6.5 ТЗ) і підготовка даних прогонів до аналізу."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricDef:
    code: str            # M1..M14
    col: str             # колонка в runs.csv / агрегованій таблиці
    name: str            # українська назва
    unit: str
    higher_better: bool
    group: str           # K1 / K2 / K3
    level: str           # run – вимірюється в кожному прогоні; method – лише на рівні методу


METRICS: list[MetricDef] = [
    MetricDef("M1", "recall", "Повнота (Recall)", "%", True, "K1", "run"),
    MetricDef("M2", "precision", "Точність вилучення (Precision)", "%", True, "K1", "run"),
    MetricDef("M3", "f1", "F1-міра", "%", True, "K1", "run"),
    MetricDef("M4", "field_accuracy", "Точність поля «ціна»", "%", True, "K1", "run"),
    MetricDef("M5", "sr", "Частка успішних прогонів (SR)", "%", True, "K1", "method"),
    MetricDef("M6", "t_ms", "Час виконання", "мс", False, "K2", "run"),
    MetricDef("M7", "throughput", "Пропускна здатність", "зап./с", True, "K2", "run"),
    MetricDef("M8", "traffic_mb", "Мережевий трафік", "МБ", False, "K2", "run"),
    MetricDef("M9", "ram_peak_mb", "Пікова пам'ять", "МБ", False, "K2", "run"),
    MetricDef("M10", "cpu_s", "Процесорний час", "с", False, "K2", "run"),
    MetricDef("M11", "fr", "Частота відмов (FR)", "%", False, "K3", "method"),
    MetricDef("M12", "antibot_hits", "Спрацювання анти-бот-захисту", "к-сть/прогін", False, "K3", "run"),
    MetricDef("M13", "ssr", "Виживання селекторів (SSR)", "%", True, "K3", "method"),
    MetricDef("M14", "effort_loc", "Трудомісткість підготовки", "LOC", False, "K3", "method"),
]
BY_COL = {m.col: m for m in METRICS}
BY_CODE = {m.code: m for m in METRICS}
RUN_METRICS = [m for m in METRICS if m.level == "run"]

# Пороги практичної значущості (п. 8.6 ТЗ). Фіксуються ДО аналізу.
#   ("abs", x)   – абсолютна різниця медіан >= x
#   ("ratio", x) – відношення більшої медіани до меншої >= x
PRACTICAL_THRESHOLDS: dict[str, tuple[str, float]] = {
    "recall": ("abs", 5.0), "precision": ("abs", 5.0), "f1": ("abs", 5.0), "field_accuracy": ("abs", 5.0),
    "t_ms": ("ratio", 1.5), "traffic_mb": ("ratio", 1.5), "ram_peak_mb": ("abs", 100.0),
    "cpu_s": ("ratio", 1.5), "throughput": ("ratio", 1.5),
    "sr": ("abs", 5.0), "fr": ("abs", 5.0), "ssr": ("abs", 5.0),
}

NUMERIC_COLS = ["n_found", "tp", "fp", "fn", "recall", "precision", "f1", "field_accuracy", "t_ms", "throughput",
                "traffic_mb", "body_mb", "ram_peak_mb", "ram_baseline_mb", "cpu_s", "antibot_hits", "n_requests",
                "n_xhr", "html_size", "status_code", "rep", "browser_launch_ms"]


def load_runs(path, block: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in NUMERIC_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["error_type"] = df.get("error_type", pd.Series(index=df.index, dtype=object)).fillna("").astype(str)
    df.loc[df["error_type"] == "nan", "error_type"] = ""
    if "success" in df:
        df["success"] = df["success"].map(lambda v: str(v).strip().lower() == "true" if str(v).strip() not in ("", "nan") else np.nan)
    if block and "block" in df:
        df = df[df["block"] == block]
    return df


def usable_for_resources(df: pd.DataFrame) -> pd.Series:
    """Прогони, придатні для часових/ресурсних показників: метод застосовано і не було мережевого збою/тайм-ауту.
    Порожній результат (empty) і 403 – це властивість методу на сторінці, тому вони НЕ виключаються."""
    return ~df["error_type"].isin(["not_applicable", "timeout", "network", "exception"])


def page_level(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Медіана повторів для кожної пари (сторінка, метод) – одиниця аналізу для критеріїв на пов'язаних вибірках."""
    keep = [c for c in cols if c in df and df[c].notna().any()]
    d = df[df["error_type"] != "not_applicable"]
    g = d.groupby(["page_id", "method"])
    out = g[keep].median()
    meta = df.groupby("page_id")[["render_type", "pagination", "site"]].first() if "pagination" in df else \
        df.groupby("page_id")[["render_type", "site"]].first()
    return out.reset_index().merge(meta.reset_index(), on="page_id", how="left")


def method_level(df: pd.DataFrame, extra: Optional[dict[str, dict[str, float]]] = None,
                 na_policy: str = "zero") -> pd.DataFrame:
    """Значення всіх доступних показників на рівні методу (для багатокритеріального оцінювання).
    Безперервні – медіана з медіан сторінок; SR/FR – частки прогонів; M12 – середня к-сть спрацювань на прогін.
    extra – {col: {method: value}} для M13, M14.

    na_policy (як враховувати сторінки, де метод НЕ застосовний, напр. М2 без endpoint):
      'zero'    – (за замовчуванням) повнота/точність/F1/пропускна здатність = 0, прогони рахуються як
                  неуспішні у SR. Метод «платить» за обмежену застосовність – чесне порівняння.
      'exclude' – такі сторінки ігноруються (показує якість методу лише там, де він працює).
    Ресурсні показники (час, пам'ять, трафік, CPU) завжди рахуються лише там, де метод виконувався."""
    quality = ["recall", "precision", "f1", "field_accuracy", "throughput"]
    resource = ["t_ms", "traffic_mb", "ram_peak_mb", "cpu_s"]
    usable = df[usable_for_resources(df)]
    res = pd.DataFrame(index=sorted(df["method"].unique()))
    for col in resource:
        if col in df and df[col].notna().any():
            pl = usable.groupby(["page_id", "method"])[col].median().unstack("method")
            res[col] = pl.median()
    for col in quality:
        if col not in df or not df[col].notna().any():
            continue
        pl = df[df["error_type"] != "not_applicable"].groupby(["page_id", "method"])[col].median().unstack("method")
        if na_policy == "zero" and col != "field_accuracy":
            na = df[df["error_type"] == "not_applicable"][["page_id", "method"]].drop_duplicates()
            for _, r in na.iterrows():
                pl.loc[r["page_id"], r["method"]] = 0.0
        res[col] = pl.median()
    runs = df if na_policy == "zero" else df[df["error_type"] != "not_applicable"]
    if "success" in runs and runs["success"].notna().any():
        sr = runs.groupby("method")["success"].apply(lambda s: 100.0 * s.fillna(False).astype(bool).mean())
        res["sr"] = sr
        res["fr"] = 100.0 - sr
    res["antibot_hits"] = df[df["error_type"] != "not_applicable"].groupby("method")["antibot_hits"].mean()
    app = df.groupby("method").apply(lambda g: 100.0 * (g["error_type"] != "not_applicable").mean(), include_groups=False)
    for col, vals in (extra or {}).items():
        res[col] = pd.Series(vals)
    res = res.dropna(axis=1, how="all")
    res.attrs["applicability_%"] = app.round(1).to_dict()
    return res


FAILURE_REASONS = {"timeout": "тайм-аут", "network": "мережева помилка", "http_403": "HTTP 403", "http_429": "HTTP 429",
                   "captcha": "CAPTCHA", "empty": "порожній результат", "parse": "виняток розбору",
                   "exception": "необроблений виняток", "other": "інше (HTTP ≥ 400)", "low_recall": "Recall < 90 %"}


def failure_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Розклад неуспішних прогонів за причинами (M11, п. 6.2 ТЗ)."""
    d = df[df["error_type"] != "not_applicable"].copy()
    reason = d["error_type"].replace("", np.nan)
    if "success" in d and d["success"].notna().any():
        low = d["success"].eq(False) & reason.isna()
        reason = reason.mask(low, "low_recall")
    d["reason"] = reason.map(lambda r: FAILURE_REASONS.get(r, r) if isinstance(r, str) else None)
    t = d.dropna(subset=["reason"]).pivot_table(index="method", columns="reason", values="run_id", aggfunc="count", fill_value=0)
    t["усього прогонів"] = d.groupby("method").size()
    return t
