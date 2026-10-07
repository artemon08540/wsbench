"""Багатокритеріальне оцінювання методів (розділ 9 ТЗ).

Порядок: 1) аналіз Парето (без ваг) -> 2) мінімакс-нормалізація до [0; 1] (1 = найкраще)
-> 3) лінійна згортка E = Σ wᵢ·xᵢ -> 4) ваги за трьома сценаріями (рівні / AHP Сааті /
прикладні профілі) -> 5) аналіз чутливості (1000 наборів ваг з відхиленням ±30 %).
E – безрозмірна величина в [0; 1], використовується лише для впорядкування методів.
"""
from __future__ import annotations

import json
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..analytics.metrics import BY_COL, METRICS

GROUPS = ["K1", "K2", "K3"]
GROUP_NAMES = {"K1": "результативність (M1–M5)", "K2": "ресурсомісткість (M6–M10)", "K3": "експлуатаційні (M11–M14)"}
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

DEFAULT_PROFILES = {
    "Повнота понад усе": {"K1": 0.70, "K2": 0.10, "K3": 0.20},
    "Масовий моніторинг": {"K1": 0.35, "K2": 0.50, "K3": 0.15},
    "Довготривала експлуатація": {"K1": 0.35, "K2": 0.20, "K3": 0.45},
}


# ---------------------------------------------------------------- підготовка
def criteria_table(method_values: pd.DataFrame) -> pd.DataFrame:
    """Лише відомі показники M1–M14, у порядку ТЗ; рядки – методи."""
    cols = [m.col for m in METRICS if m.col in method_values.columns and method_values[m.col].notna().any()]
    return method_values[cols].copy()


def drop_constant(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Показник, однаковий для всіх методів, не розрізняє їх: нормалізація дала б 0/0."""
    const = [c for c in t.columns if t[c].nunique(dropna=True) <= 1]
    return t.drop(columns=const), const


# ---------------------------------------------------------------- 9.2 Парето
def dominates(a: pd.Series, b: pd.Series) -> bool:
    better_or_equal, strictly = True, False
    for col in a.index:
        hb = BY_COL[col].higher_better
        x, y = a[col], b[col]
        if pd.isna(x) or pd.isna(y):
            continue
        if (x < y) if hb else (x > y):
            better_or_equal = False
            break
        if (x > y) if hb else (x < y):
            strictly = True
    return better_or_equal and strictly


def pareto(t: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Матриця домінування («рядок домінує над стовпцем») і множина Парето-оптимальних методів."""
    methods = list(t.index)
    dom = pd.DataFrame("", index=methods, columns=methods)
    for a in methods:
        for b in methods:
            if a != b and dominates(t.loc[a], t.loc[b]):
                dom.loc[a, b] = "домінує"
    front = [m for m in methods if not (dom[m] == "домінує").any()]
    return dom, front


# ---------------------------------------------------------------- 9.3 нормалізація
def normalize(t: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index)
    for c in t.columns:
        x = t[c].astype(float)
        lo, hi = x.min(), x.max()
        if hi == lo:
            out[c] = 1.0
            continue
        out[c] = (x - lo) / (hi - lo) if BY_COL[c].higher_better else (hi - x) / (hi - lo)
    return out


# ---------------------------------------------------------------- 9.4 згортка
def metric_weights_from_groups(cols: list[str], group_w: dict[str, float]) -> pd.Series:
    """Вага групи ділиться порівну між її показниками, присутніми в даних (п. 9.5, крок 5).
    Якщо в групі немає жодного показника – її вага перерозподіляється між іншими групами."""
    present = {g: [c for c in cols if BY_COL[c].group == g] for g in GROUPS}
    gw = {g: group_w.get(g, 0.0) for g in GROUPS if present[g]}
    s = sum(gw.values())
    w = {}
    for g, wg in gw.items():
        for c in present[g]:
            w[c] = (wg / s) / len(present[g])
    return pd.Series(w)[cols]


def weighted_sum(norm: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(norm.columns).fillna(0)
    w = w / w.sum()
    return (norm * w).sum(axis=1).rename("E")


# ---------------------------------------------------------------- 9.5 AHP
@dataclass
class AhpResult:
    weights: dict[str, float]
    lambda_max: float
    ci: float
    cr: float
    consistent: bool
    matrix: list[list[float]]


def ahp(matrix: list[list[float]], labels: Optional[list[str]] = None) -> AhpResult:
    a = np.array(matrix, dtype=float)
    n = a.shape[0]
    if a.shape != (n, n) or not np.allclose(np.diag(a), 1) or not np.allclose(a * a.T, 1, rtol=1e-2):
        raise ValueError("Матриця AHP має бути квадратною, з одиницями на діагоналі й обернено-симетричною (aji = 1/aij)")
    vals, vecs = np.linalg.eig(a)
    i = int(np.argmax(vals.real))
    w = np.abs(vecs[:, i].real)
    w = w / w.sum()
    lam = float(vals[i].real)
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri else 0.0
    labels = labels or GROUPS[:n]
    return AhpResult({l: float(x) for l, x in zip(labels, w)}, lam, ci, cr, cr < 0.1, a.tolist())


def load_ahp(path: Path) -> AhpResult:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    # дозволяємо записувати дроби рядками: "1/3"
    m = [[float(Fraction(v)) if isinstance(v, str) else float(v) for v in row] for row in cfg["matrix"]]
    return ahp(m, cfg.get("criteria"))


# ---------------------------------------------------------------- 9.6 чутливість
def sensitivity(norm: pd.DataFrame, base_group_w: dict[str, float], n: int = 1000, spread: float = 0.30,
                seed: int = 42, level: str = "metric") -> pd.DataFrame:
    """1000 випадкових наборів ваг: кожна вага множиться на U(1−spread; 1+spread), далі нормалізація до суми 1.
    level='metric' – збурюються ваги окремих показників; 'group' – ваги груп К1–К3."""
    rng = np.random.default_rng(seed)
    base = metric_weights_from_groups(list(norm.columns), base_group_w)
    wins = {m: 0 for m in norm.index}
    for _ in range(n):
        if level == "group":
            gw = {g: v * rng.uniform(1 - spread, 1 + spread) for g, v in base_group_w.items()}
            w = metric_weights_from_groups(list(norm.columns), gw)
        else:
            w = base * rng.uniform(1 - spread, 1 + spread, size=len(base))
        e = weighted_sum(norm, w)
        wins[e.idxmax()] += 1
    return pd.DataFrame({"Перемог": wins, "Частка перемог, %": {m: 100 * v / n for m, v in wins.items()}})


def e_vs_k1(norm: pd.DataFrame, base_group_w: dict[str, float], steps: int = 21) -> pd.DataFrame:
    """E кожного методу при зміні ваги К1 від 0 до 1 (решта груп – пропорційно базовим вагам)."""
    rest = {g: v for g, v in base_group_w.items() if g != "K1"}
    s = sum(rest.values()) or 1.0
    rows = []
    for w1 in np.linspace(0, 1, steps):
        gw = {"K1": w1, **{g: (1 - w1) * v / s for g, v in rest.items()}}
        e = weighted_sum(norm, metric_weights_from_groups(list(norm.columns), gw))
        rows.append({"w_K1": round(float(w1), 3), **e.to_dict()})
    return pd.DataFrame(rows).set_index("w_K1")


# ---------------------------------------------------------------- усе разом
def evaluate_all(method_values: pd.DataFrame, ahp_res: Optional[AhpResult] = None,
                 profiles: Optional[dict] = None) -> dict:
    t = criteria_table(method_values)
    t, const = drop_constant(t)
    dom, front = pareto(t)
    norm = normalize(t)
    cols = list(norm.columns)
    scenarios = {"Рівні ваги": {g: 1 / 3 for g in GROUPS}}
    if ahp_res:
        scenarios["AHP (Сааті)"] = ahp_res.weights
    scenarios.update(profiles or DEFAULT_PROFILES)
    e_table = pd.DataFrame({name: weighted_sum(norm, metric_weights_from_groups(cols, gw)) for name, gw in scenarios.items()})
    ranks = e_table.rank(ascending=False, method="min").astype(int)
    base = ahp_res.weights if ahp_res else scenarios["Рівні ваги"]
    return {
        "criteria": t, "constant_dropped": const, "dominance": dom, "pareto_front": front, "normalized": norm,
        "scenarios": scenarios, "E": e_table, "ranks": ranks,
        "sensitivity": sensitivity(norm, base), "sensitivity_groups": sensitivity(norm, base, level="group"),
        "e_vs_k1": e_vs_k1(norm, base), "ahp": ahp_res,
    }
