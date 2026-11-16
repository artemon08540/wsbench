"""Відтворення результатів М4 з новим критерієм перемикання за журналом основного експерименту.

Чому так, а не перезапуск. Рішення М4 – детермінована функція відповіді М1 (n_M1, частка карток із
назвою і ціною, кнопка «Показати ще» в HTML). Після рішення М4 виконує або лише М1, або М1, а потім М3.
Обидва кроки виміряно в основному експерименті 10.11.2026 у тих самих умовах і в тих самих повторах.
Повний перезапуск 16.11 був неможливий: еталон застарів (асортимент змінився), а Telemart став
відповідати 403 усім методам (results/recheck_2026-11-16).

Правила відтворення прогону М4 (сторінка p, повтор r):
  • рішення нового критерію збігається з рішенням, яке фактично прийняв М4 10.11 → прогін М4 залишається;
  • новий критерій НЕ запускає браузер, а 10.11 М4 його запускав → береться прогін М1 (p, r);
  • новий критерій запускає браузер, а 10.11 М4 його не запускав → якість – з прогону М3 (p, r),
    витрати – сума М1 + М3 (час, трафік, запити, процесорний час), пам'ять – пік М3 над базою М1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

QUALITY = ["n_found", "tp", "fp", "fn", "recall", "precision", "f1", "field_accuracy", "n_fuzzy", "success",
           "status_code", "error_type", "error_msg"]
SUMMED = ["t_ms", "traffic_mb", "body_mb", "cpu_s", "n_requests", "antibot_hits"]


def load_signals(path: Path) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["pages"]


def decide(sig: dict, threshold: float, min_records: int = 5) -> str:
    if sig["n_m1"] < min_records:
        return "few_records"
    if sig["fill"] < threshold:
        return "low_fill"
    if sig["load_more"]:
        return "load_more"
    return ""


def reconstruct(df: pd.DataFrame, signals: dict[str, dict], threshold: float = 0.8) -> pd.DataFrame:
    """Повертає журнал, у якому прогони М4 замінено відтвореними за новим критерієм."""
    df = df.copy()
    key = ["page_id", "rep"]
    m1 = df[df["method"] == "M1"].set_index(key)
    m3 = df[df["method"] == "M3"].set_index(key)
    rows = []
    for _, r in df[df["method"] == "M4"].iterrows():
        sig = signals[r["page_id"]]
        reason = decide(sig, threshold)
        was = str(r.get("fallback")).strip().lower() == "true"
        k = (r["page_id"], r["rep"])
        new = r.copy()
        if bool(reason) == was:
            new["reconstructed"] = False
        elif not reason:                       # браузер більше не запускається → прогін М1
            src = m1.loc[k]
            for c in QUALITY + SUMMED + ["ram_peak_mb", "ram_baseline_mb", "html_size", "n_xhr", "throughput"]:
                if c in src:
                    new[c] = src[c]
            new["fallback"], new["reconstructed"] = False, True
        else:                                  # браузер тепер запускається → М1 + М3
            a, b = m1.loc[k], m3.loc[k]
            for c in QUALITY:
                if c in b:
                    new[c] = b[c]
            for c in SUMMED:
                if c in a and c in b:
                    new[c] = (0 if pd.isna(a[c]) else a[c]) + (0 if pd.isna(b[c]) else b[c])
            new["ram_baseline_mb"] = a["ram_baseline_mb"]
            new["ram_peak_mb"] = max(a["ram_peak_mb"], a["ram_baseline_mb"] + (b["ram_peak_mb"] - b["ram_baseline_mb"]))
            new["throughput"] = (new["n_found"] / (new["t_ms"] / 1000)) if new["t_ms"] else np.nan
            new["html_size"] = a.get("html_size")
            new["fallback"], new["reconstructed"] = True, True
        new["m1_found"], new["m1_fill"], new["m1_load_more"] = sig["n_m1"], sig["fill"], sig["load_more"]
        new["switch_reason"], new["threshold"] = reason, threshold
        rows.append(new)
    out = pd.concat([df[df["method"] != "M4"], pd.DataFrame(rows)], ignore_index=True)
    out["reconstructed"] = out.get("reconstructed", pd.Series(False, index=out.index)).fillna(False)
    return out.sort_values("order_idx" if "order_idx" in out else "run_id").reset_index(drop=True)


def threshold_table(df: pd.DataFrame, signals: dict[str, dict], thresholds=(0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    from .metrics import load_runs  # noqa: F401  (ram_delta обчислюється тут же)
    rows = []
    for th in thresholds:
        r = reconstruct(df, signals, th)
        m4 = r[r["method"] == "M4"].copy()
        m4["ram_delta_mb"] = m4["ram_peak_mb"] - m4["ram_baseline_mb"]
        sw = sorted({p for p, s in signals.items() if decide(s, th)})
        pl = m4.groupby("page_id")[["recall", "field_accuracy", "t_ms", "traffic_mb", "ram_delta_mb", "cpu_s"]].median()
        rows.append({"Поріг θ": th, "Сторінок з браузером": len(sw), "із 22": len(signals),
                     "Повнота, % (медіана)": pl["recall"].median(), "Повнота, % (середнє)": pl["recall"].mean(),
                     "Точність ціни, % (середнє)": pl["field_accuracy"].mean(),
                     "Час, мс (медіана)": pl["t_ms"].median(), "Трафік, МБ (медіана)": pl["traffic_mb"].median(),
                     "Пам'ять, МБ (медіана)": pl["ram_delta_mb"].median(), "CPU, с (медіана)": pl["cpu_s"].median(),
                     "Відтворено прогонів": int(m4["reconstructed"].sum())})
    return pd.DataFrame(rows)
