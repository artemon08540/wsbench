"""Статистичне оброблення результатів (розділ 8 ТЗ).

Порядок: описова статистика -> викиди -> нормальність -> Фрідман -> попарний Вілкоксон з
поправкою Холма -> розмір ефекту (Cliff's delta) -> практична значущість -> порівняння груп
(Манн–Вітні) -> кореляції Спірмена. Усі функції повертають pandas.DataFrame, готові до таблиць.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import BY_COL, PRACTICAL_THRESHOLDS, page_level, usable_for_resources

ALPHA = 0.05
BOOTSTRAP_N = 10_000
SEED = 42


# ---------------------------------------------------------------- допоміжне
def fmt_p(p: float) -> str:
    """Правило оформлення p (п. 8.7): p < 0,001 – саме так, решта – три знаки."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def holm(pvals: Iterable[float]) -> np.ndarray:
    """Поправка Холма–Бонферроні (еквівалент statsmodels multipletests(method='holm'))."""
    p = np.asarray(list(pvals), dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)          # монотонність
        adj[idx] = running
    return adj


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    """δ = (#(a>b) − #(a<b)) / (n_a·n_b). Додатне δ – значення A частіше більші за B."""
    a = np.asarray([x for x in a if not pd.isna(x)], dtype=float)
    b = np.asarray([x for x in b if not pd.isna(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def delta_label(d: float) -> str:
    if pd.isna(d):
        return "—"
    d = abs(d)
    return "мізерний" if d < 0.147 else "малий" if d < 0.33 else "середній" if d < 0.474 else "великий"


def rho_label(r: float) -> str:
    if pd.isna(r):
        return "—"
    r = abs(r)
    return ("дуже слабкий" if r < 0.2 else "слабкий" if r < 0.4 else "помірний" if r < 0.6
            else "сильний" if r < 0.8 else "дуже сильний")


def bootstrap_median_ci(x: Iterable[float], n: int = BOOTSTRAP_N, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray([v for v in x if not pd.isna(v)], dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        v = float(x[0]) if len(x) else np.nan
        return v, v
    res = stats.bootstrap((x,), np.median, n_resamples=n, confidence_level=level, method="percentile",
                          random_state=np.random.default_rng(SEED))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _data_for(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Для ресурсних показників виключаємо мережеві збої/тайм-аути; для якісних – беремо все застосоване."""
    resource = metric in ("t_ms", "throughput", "traffic_mb", "ram_delta_mb", "cpu_s")
    d = df[usable_for_resources(df)] if resource else df[df["error_type"] != "not_applicable"]
    return d[d[metric].notna()] if metric in d else d.iloc[0:0]


# ---------------------------------------------------------------- 8.1 описова статистика
def descriptive(df: pd.DataFrame, metric: str, with_ci: bool = True) -> pd.DataFrame:
    d = _data_for(df, metric)
    rows = []
    for m, g in d.groupby("method"):
        x = g[metric].astype(float)
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        mean = x.mean()
        row = {"Метод": m, "n": len(x), "Медіана": x.median(), "Середнє": mean, "SD": x.std(ddof=1),
               "IQR": q3 - q1, "Q1": q1, "Q3": q3, "Min": x.min(), "Max": x.max(),
               "CV, %": (x.std(ddof=1) / mean * 100) if mean else np.nan}
        if with_ci:
            row["95% ДІ медіани"] = "[{:.2f}; {:.2f}]".format(*bootstrap_median_ci(x))
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def describe_all(df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    return {m: descriptive(df, m) for m in metrics if m in df and df[m].notna().any()}


# ---------------------------------------------------------------- 8.2 викиди
def outliers(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Правило 1,5·IQR у межах кожної пари (сторінка, метод). Викиди НЕ видаляються – лише позначаються."""
    d = _data_for(df, metric)
    out = []
    for (pid, m), g in d.groupby(["page_id", "method"]):
        x = g[metric]
        if len(x) < 4:
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
        else:
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bad = g[(x < lo) | (x > hi)]
        for _, r in bad.iterrows():
            out.append({"run_id": r["run_id"], "page_id": pid, "method": m, "metric": metric,
                        "value": r[metric], "low": lo, "high": hi, "error_type": r.get("error_type", "")})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 8.3 нормальність
def normality(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Шапіро–Вілк на рівні сторінок (медіана повторів) для кожного методу."""
    pl = page_level(df, metrics)
    rows = []
    for metric in metrics:
        if metric not in pl:
            continue
        for m, g in pl.groupby("method"):
            x = g[metric].dropna()
            if len(x) < 3 or x.nunique() < 3:
                rows.append({"Показник": metric, "Метод": m, "n": len(x), "W": np.nan, "p": np.nan,
                             "Нормальний?": "не визначено (мало різних значень)"})
                continue
            w, p = stats.shapiro(x)
            rows.append({"Показник": metric, "Метод": m, "n": len(x), "W": round(w, 4), "p": p,
                         "Нормальний?": "так" if p >= ALPHA else "ні"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 8.4 Фрідман
def _blocks(df: pd.DataFrame, metric: str, methods: Optional[list[str]] = None, na_policy: str = "exclude") -> pd.DataFrame:
    """Матриця «сторінка × метод» (медіана повторів).
    na_policy='exclude' – сторінки, де хоч один метод не застосовний, виключаються (повні блоки);
    na_policy='zero'    – для показників повноти неприйнятий метод отримує 0 (М2 без endpoint зібрав 0)."""
    d = _data_for(df, metric)
    wide = d.groupby(["page_id", "method"])[metric].median().unstack("method")
    if methods:
        wide = wide.reindex(columns=[m for m in methods if m in wide.columns])
    if na_policy == "zero" and metric in ("recall", "precision", "f1", "n_found", "tp"):
        na = df[df["error_type"] == "not_applicable"][["page_id", "method"]].drop_duplicates()
        for _, r in na.iterrows():
            if r["method"] in wide.columns:
                wide.loc[r["page_id"], r["method"]] = 0.0
    return wide.dropna()


def friedman(df: pd.DataFrame, metric: str, methods: Optional[list[str]] = None, na_policy: str = "exclude") -> dict:
    wide = _blocks(df, metric, methods, na_policy)
    k, n = wide.shape[1], wide.shape[0]
    res = {"Показник": metric, "N (сторінок)": n, "k (методів)": k, "Методи": ", ".join(wide.columns)}
    if k < 3 or n < 2:
        res.update({"χ²": np.nan, "p": np.nan, "W Кендалла": np.nan, "Примітка": "потрібно ≥3 методи і ≥2 сторінки"})
        return res
    if np.allclose(wide.values, wide.values[:, :1]):
        res.update({"χ²": 0.0, "p": 1.0, "W Кендалла": 0.0, "Примітка": "усі методи однакові на всіх сторінках"})
        return res
    chi2, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    res.update({"χ²": round(float(chi2), 3), "p": float(p), "W Кендалла": round(float(chi2) / (n * (k - 1)), 3),
                "Примітка": ""})
    res["Середні ранги"] = "; ".join(f"{c}: {v:.2f}" for c, v in wide.rank(axis=1).mean().items())
    return res


# ---------------------------------------------------------------- 8.5–8.7 попарні порівняння і таблиця значущості
def _practical(metric: str, a: float, b: float) -> Optional[bool]:
    rule = PRACTICAL_THRESHOLDS.get(metric)
    if rule is None or pd.isna(a) or pd.isna(b):
        return None
    kind, thr = rule
    if kind == "abs":
        return abs(a - b) >= thr
    lo, hi = sorted([abs(a), abs(b)])
    return (hi / lo >= thr) if lo > 0 else hi > 0


def significance_table(df: pd.DataFrame, metric: str, methods: Optional[list[str]] = None,
                       na_policy: str = "exclude", min_delta: float = 0.33) -> pd.DataFrame:
    """«Таблиця значущості» (п. 8.7): медіани, різниця, p (сире і за Холмом), Cliff's δ, суттєвість.
    Суттєво = p_Холм < 0,05  І  |δ| ≥ 0,33  І  перевищено поріг практичної значущості."""
    wide = _blocks(df, metric, methods, na_policy)
    rows = []
    for a, b in combinations(wide.columns, 2):
        x, y = wide[a].values, wide[b].values
        diff = y - x
        if np.allclose(diff, 0):
            p = 1.0
        else:
            try:
                p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
        ma, mb = float(np.median(x)), float(np.median(y))
        d = cliffs_delta(x, y)
        rows.append({"Пара": f"{a} vs {b}", "Медіана A": ma, "Медіана B": mb, "Різниця (A−B)": ma - mb,
                     "Відношення A/B": (ma / mb) if mb else np.nan, "p (сире)": p, "δ": d,
                     "Ефект": delta_label(d), "Практ. поріг": _practical(metric, ma, mb), "N": len(wide)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p (Холм)"] = holm(t["p (сире)"])
    t["Суттєво?"] = [
        "так" if (ph < ALPHA and abs(d) >= min_delta and (pr is None or pr)) else "ні"
        for ph, d, pr in zip(t["p (Холм)"], t["δ"], t["Практ. поріг"])
    ]
    cols = ["Пара", "Медіана A", "Медіана B", "Різниця (A−B)", "Відношення A/B", "p (сире)", "p (Холм)", "δ",
            "Ефект", "Практ. поріг", "Суттєво?", "N"]
    return t[cols]


def holm_across(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Додаткова (суворіша) поправка Холма на ВСІ порівняння всіх показників разом (14 × 6 = 84 тести)."""
    keys, ps = [], []
    for m, t in tables.items():
        for i, p in enumerate(t.get("p (сире)", [])):
            keys.append((m, i))
            ps.append(p)
    adj = holm(ps)
    out = {m: t.copy() for m, t in tables.items()}
    for (m, i), pa in zip(keys, adj):
        out[m].loc[out[m].index[i], "p (Холм, усі показники)"] = pa
    return out


# ---------------------------------------------------------------- 8.8 групи ресурсів і кореляції
def gap_by_group(df: pd.DataFrame, metric: str = "recall", method_hi: str = "M3", method_lo: str = "M1",
                 group_col: str = "render_type", g1: str = "spa", g2: str = "ssr") -> dict:
    """Друга частина гіпотези 1: розрив (M3 − M1) у групі SPA більший, ніж у SSR? Манн–Вітні U (незалежні групи)."""
    pl = page_level(df, [metric])
    wide = pl.pivot_table(index=["page_id", group_col], columns="method", values=metric).reset_index()
    if method_hi not in wide or method_lo not in wide:
        return {"Примітка": "немає потрібних методів"}
    wide["gap"] = wide[method_hi] - wide[method_lo]
    a = wide.loc[wide[group_col] == g1, "gap"].dropna()
    b = wide.loc[wide[group_col] == g2, "gap"].dropna()
    res = {"Показник": metric, "Розрив": f"{method_hi} − {method_lo}", "Група 1": g1, "n1": len(a),
           "Медіана розриву 1": a.median(), "Група 2": g2, "n2": len(b), "Медіана розриву 2": b.median()}
    if len(a) < 2 or len(b) < 2:
        res.update({"U": np.nan, "p (однобічний, 1 > 2)": np.nan, "δ": np.nan, "Примітка": "замало сторінок у групі"})
        return res
    if np.allclose(np.concatenate([a, b]), a.iloc[0]):
        u, p = np.nan, 1.0
    else:
        u, p = stats.mannwhitneyu(a, b, alternative="greater")
    d = cliffs_delta(a, b)
    res.update({"U": u, "p (однобічний, 1 > 2)": float(p), "δ": d, "Ефект": delta_label(d), "Примітка": ""})
    return res


DEFAULT_CORRELATIONS = [
    # (характеристика сторінки, метод-джерело характеристики, показник, метод показника, опис)
    ("n_found", "M3", "t_ms", None, "кількість товарів ↔ час виконання"),
    ("html_size", "M1", "recall", "M1", "обсяг початкового HTML ↔ повнота М1"),
    ("n_xhr", "M3", "t_ms", "M3", "кількість XHR-запитів ↔ час М3"),
    ("n_requests", "M3", "traffic_mb", "M3", "кількість запитів ↔ трафік М3"),
]


def correlations(df: pd.DataFrame, pairs=None) -> pd.DataFrame:
    """Спірмен між характеристиками сторінки і показниками методів (на рівні сторінок)."""
    pairs = pairs or DEFAULT_CORRELATIONS
    pl = page_level(df[usable_for_resources(df)], ["n_found", "html_size", "n_xhr", "n_requests", "t_ms",
                                                    "recall", "traffic_mb", "ram_delta_mb"])
    rows = []
    for feat, feat_m, metric, metric_m, label in pairs:
        if feat not in pl or metric not in pl:
            continue
        f = pl[pl["method"] == feat_m].set_index("page_id")[feat]
        targets = [metric_m] if metric_m else sorted(pl["method"].unique())
        for tm in targets:
            y = pl[pl["method"] == tm].set_index("page_id")[metric]
            j = pd.concat([f, y], axis=1, keys=["x", "y"]).dropna()
            if len(j) < 3 or j["x"].nunique() < 2 or j["y"].nunique() < 2:
                continue
            r, p = stats.spearmanr(j["x"], j["y"])
            rows.append({"Зв'язок": label, "Метод": tm, "n": len(j), "ρ": round(float(r), 3), "p": float(p),
                         "Сила": rho_label(r), "Напрямок": "прямий" if r > 0 else "обернений"})
    return pd.DataFrame(rows)


def metric_redundancy(df: pd.DataFrame, cols: list[str], threshold: float = 0.9) -> pd.DataFrame:
    """Перевірка надлишковості системи показників (п. 6.5): пари з |ρ| > 0,9."""
    pl = page_level(df[usable_for_resources(df)], cols)
    present = [c for c in cols if c in pl and pl[c].nunique() > 1]
    corr = pl[present].corr(method="spearman")
    rows = [{"Показник 1": a, "Показник 2": b, "ρ": round(corr.loc[a, b], 3)}
            for a, b in combinations(present, 2) if abs(corr.loc[a, b]) > threshold]
    return pd.DataFrame(rows)


def label_metric(col: str) -> str:
    m = BY_COL.get(col)
    return f"{m.code} {m.name}, {m.unit}" if m else col


# ---------------------------------------------------------------- гіпотеза 3: порівняння часток (M13)
def survival_tests(surv: pd.DataFrame, focus: str = "M2") -> pd.DataFrame:
    """Точний критерій Фішера (2×2: працює / не працює) для пар методів + χ² по всіх методах.
    Гіпотеза 3: метод на JSON-API (М2) має вищу частку виживання, ніж методи на CSS-селекторах."""
    d = surv[(~surv["is_reference"]) & surv["works"].notna()]
    counts = d.groupby("method")["works"].agg(["sum", "size"])
    rows = []
    others = [m for m in counts.index if m != focus] if focus in counts.index else []
    pairs = [(focus, o) for o in others] or list(combinations(counts.index, 2))
    for a, b in pairs:
        t = [[int(counts.loc[a, "sum"]), int(counts.loc[a, "size"] - counts.loc[a, "sum"])],
             [int(counts.loc[b, "sum"]), int(counts.loc[b, "size"] - counts.loc[b, "sum"])]]
        _, p_two = stats.fisher_exact(t)
        _, p_one = stats.fisher_exact(t, alternative="greater")
        rows.append({"Пара": f"{a} vs {b}", "SSR A, %": 100 * t[0][0] / max(1, sum(t[0])),
                     "SSR B, %": 100 * t[1][0] / max(1, sum(t[1])), "p Фішера (двобічний)": p_two,
                     "p Фішера (A > B)": p_one})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p (Холм, A > B)"] = holm(res["p Фішера (A > B)"])
    table = [[int(r["sum"]), int(r["size"] - r["sum"])] for _, r in counts.iterrows()]
    if len(table) >= 2 and all(sum(x) > 0 for x in zip(*table)):
        chi2, p, dof, _ = stats.chi2_contingency(table)
        res.attrs["chi2"] = {"χ²": round(float(chi2), 3), "df": int(dof), "p": float(p)}
    return res


# ---------------------------------------------------------------- рівень магазинів
def site_level(df: pd.DataFrame, metrics=("recall", "field_accuracy", "t_ms", "traffic_mb", "ram_delta_mb", "cpu_s")) -> pd.DataFrame:
    """Магазин × метод: медіана з медіан сторінок магазину. Сторінки одного магазину мають спільну
    верстку й селектори, тому незалежною одиницею є магазин (10 спостережень замість 22)."""
    metrics = [m for m in metrics if m in df and df[m].notna().any()]
    pl = page_level(df[usable_for_resources(df) | df["error_type"].isin(["empty"])], metrics)
    meta = pl.groupby("site").agg(сторінок=("page_id", "nunique"),
                                  тип=("render_type", lambda s: "/".join(sorted(set(s)))))
    t = pl.groupby(["site", "method"])[metrics].median().unstack("method")
    t.columns = [f"{m} · {meth}" for m, meth in t.columns]
    return meta.join(t)


def site_level_tests(df: pd.DataFrame, metric: str = "recall", pairs=(("M1", "M3"), ("M1", "M4"), ("M3", "M4"))) -> pd.DataFrame:
    """Вілкоксон + Холм на рівні магазинів (N = кількість магазинів, де застосовні обидва методи)."""
    pl = page_level(df, [metric])
    w = pl.groupby(["site", "method"])[metric].median().unstack("method")
    rows = []
    for a, b in pairs:
        if a not in w or b not in w:
            continue
        d = w[[a, b]].dropna()
        x, y = d[a].to_numpy(float), d[b].to_numpy(float)
        if len(d) < 2 or np.allclose(x, y):
            p = 1.0
        else:
            try:
                p = float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)
            except ValueError:
                p = 1.0
        rows.append({"Показник": metric, "Пара": f"{a} vs {b}", "N магазинів": len(d), "Медіана A": np.median(x),
                     "Медіана B": np.median(y), "p": p, "δ": cliffs_delta(x, y)})
    t = pd.DataFrame(rows)
    if len(t):
        t["p (Холм)"] = holm(t["p"].tolist())
        t["Ефект"] = t["δ"].map(delta_label)
    return t


# ---------------------------------------------------------------- чутливість М4 до порогу
def m4_threshold_sensitivity(df: pd.DataFrame, thresholds=(0.6, 0.7, 0.8, 0.9), min_records: int = 5) -> pd.DataFrame:
    """Рішення М4 – детермінована функція відповіді М1 (n_M1, m1_fill, m1_load_more), тому результат при
    іншому порозі відтворюється з журналу: якщо при порозі θ рішення збігається з фактичним – береться
    фактичний прогін М4; якщо ні – прогін М1 (браузер не запускався б) або М3 (запускався б) того ж повтору."""
    m4 = df[df["method"] == "M4"].copy()
    if "m1_fill" not in m4 or m4["m1_fill"].isna().all():
        return pd.DataFrame()
    m4["m1_fill"] = pd.to_numeric(m4["m1_fill"], errors="coerce").fillna(0)
    m4["m1_found"] = pd.to_numeric(m4["m1_found"], errors="coerce").fillna(0)
    lm = m4["m1_load_more"].astype(str).str.lower().eq("true")
    actual = m4["fallback"].astype(str).str.lower().eq("true")
    blocked = m4.get("switch_reason", pd.Series("", index=m4.index)).astype(str).eq("blocked")
    key = ["page_id", "rep"]
    alt = {m: df[df["method"] == m].set_index(key) for m in ("M1", "M3")}
    rows = []
    for th in thresholds:
        sw = ~blocked & ((m4["m1_found"] < min_records) | (m4["m1_fill"] < th) | lm)
        rec, t, n_sw = [], [], int(sw.sum())
        for i, r in m4.iterrows():
            if sw[i] == actual[i]:
                src = r
            else:
                k = (r["page_id"], r["rep"])
                src = alt["M3" if sw[i] else "M1"].loc[k] if k in alt["M3" if sw[i] else "M1"].index else r
            rec.append(src["recall"]), t.append(src["t_ms"])
        rows.append({"Поріг θ": th, "Перемикань на браузер": n_sw, "Частка, %": 100 * n_sw / len(m4),
                     "Повнота (медіана), %": float(np.nanmedian(rec)), "Повнота (середнє), %": float(np.nanmean(rec)),
                     "Час (медіана), мс": float(np.nanmedian(t)), "Відрізняється від фактичного": int((sw != actual).sum())})
    return pd.DataFrame(rows)
