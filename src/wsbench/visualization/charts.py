"""Графіки для розділу 3 (matplotlib, PNG 200 dpi – придатні для друку).

Колір несе лише ідентичність методу і завжди один і той самий (M1 синій, M2 помаранчевий,
M3 бірюзовий, M4 жовтий – фіксований порядок, не залежить від того, скільки методів на графіку).
Для друку в ч/б і для людей з порушеннями кольоросприйняття кожен метод має ще й власну форму
маркера / штрихування, а ряди підписані безпосередньо.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..analytics.metrics import BY_COL, page_level, usable_for_resources  # noqa: E402

COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100"}
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D"}
HATCH = {"M1": "", "M2": "//", "M3": "..", "M4": "xx"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHOD_NAMES = {"M1": "М1 HTTP+HTML", "M2": "М2 внутрішній API", "M3": "М3 headless", "M4": "М4 гібрид"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _label(col: str) -> str:
    m = BY_COL.get(col)
    return f"{m.name}, {m.unit}" if m else col


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Розсуває вертикальні позиції підписів, щоб вони не накладались (зберігаючи порядок)."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for k in range(1, len(order)):
        prev, cur = order[k - 1], order[k]
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out.tolist()


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def boxplot(df: pd.DataFrame, metric: str, path: Path, log: bool = False) -> Path:
    """Діаграма розмаху показника за методами (рівень прогонів) – п. 8.2 ТЗ."""
    d = df[usable_for_resources(df)] if metric in ("t_ms", "traffic_mb", "ram_peak_mb", "cpu_s", "throughput") \
        else df[df["error_type"] != "not_applicable"]
    methods = [m for m in COLORS if m in d["method"].unique()]
    data = [d.loc[d["method"] == m, metric].dropna().values for m in methods]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK, "linewidth": 1.6},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK2, "markeredgecolor": "none", "alpha": .6},
                    whiskerprops={"color": INK2}, capprops={"color": INK2})
    for patch, m in zip(bp["boxes"], methods):
        patch.set(facecolor=COLORS[m], alpha=.85, edgecolor="white", linewidth=1.5, hatch=HATCH[m])
    ax.set_xticks(range(1, len(methods) + 1), [METHOD_NAMES[m] for m in methods])
    ax.set_xlim(0.5, len(methods) + 0.85)
    for i, x in enumerate(data, start=1):
        if len(x):
            ax.annotate(f"{np.median(x):,.1f}".replace(",", " "), (i + 0.27, np.median(x)), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=8, color=INK,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": .85})
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(_label(metric) + (" (лог. шкала)" if log else ""))
    ax.set_title(f"{BY_COL[metric].name if metric in BY_COL else metric} (показник {BY_COL[metric].code if metric in BY_COL else ''}): розподіл за методами")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def recall_by_render(df: pd.DataFrame, path: Path) -> Path:
    """Гіпотеза 1 наочно: медіанна повнота кожного методу в групах SSR / гібрид / SPA."""
    pl = page_level(df, ["recall"])
    order = [g for g in ["ssr", "hybrid", "spa"] if g in pl["render_type"].unique()]
    methods = [m for m in COLORS if m in pl["method"].unique()]
    med = pl.groupby(["render_type", "method"])["recall"].median().unstack("method").reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(methods)
    x = np.arange(len(order))
    for i, m in enumerate(methods):
        vals = med[m].values
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w * 0.92, color=COLORS[m], hatch=HATCH[m],
                      edgecolor="white", linewidth=1.5, label=METHOD_NAMES[m])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [{"ssr": "SSR", "hybrid": "Гібрид", "spa": "SPA / CSR"}[g] for g in order])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Повнота (медіана по сторінках), %")
    ax.set_title("Повнота (показник M1) за типом рендерингу сторінки")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    return _save(fig, path)


def scatter_recall_time(df: pd.DataFrame, path: Path) -> Path:
    """Компроміс «повнота – час»: кожна точка – пара (сторінка, метод), медіана повторів."""
    pl = page_level(df[usable_for_resources(df)], ["recall", "t_ms"]).dropna(subset=["recall", "t_ms"])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for m in [m for m in COLORS if m in pl["method"].unique()]:
        g = pl[pl["method"] == m]
        ax.scatter(g["t_ms"], g["recall"], s=42, marker=MARKERS[m], color=COLORS[m], edgecolor="white", linewidth=1.2,
                   alpha=.9, label=METHOD_NAMES[m], zorder=3)
        ax.annotate(METHOD_NAMES[m].split()[0], (g["t_ms"].median(), g["recall"].median()), xytext=(6, 6),
                    textcoords="offset points", fontsize=8.5, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Час виконання, мс (лог. шкала)")
    ax.set_ylabel("Повнота, %")
    ax.set_ylim(-5, 108)
    ax.set_title("Повнота vs час: кожна точка – сторінка × метод")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8.5)
    return _save(fig, path)


def corr_heatmap(df: pd.DataFrame, cols: list[str], path: Path) -> Path:
    """Теплова карта кореляцій Спірмена між показниками (перевірка надлишковості, п. 6.5)."""
    pl = page_level(df[usable_for_resources(df)], cols)
    present = [c for c in cols if c in pl and pl[c].nunique() > 1]
    corr = pl[present].corr(method="spearman")
    labels = [BY_COL[c].code if c in BY_COL else c for c in present]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(present)), labels)
    ax.set_yticks(range(len(present)), labels)
    ax.grid(False)
    for i in range(len(present)):
        for j in range(len(present)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > .6 else INK)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="ρ Спірмена")
    ax.set_title("Кореляції між показниками")
    return _save(fig, path)


def pareto_chart(crit: pd.DataFrame, front: list[str], path: Path, x: str = "t_ms", y: str = "recall") -> Path:
    """Проєкція аналізу Парето на дві осі (повнота vs час); недоміновані методи обведено."""
    if x not in crit or y not in crit:
        return path
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ys = list(crit[y].astype(float))
    span = (max(ys) - min(ys)) or 1.0
    label_y = _spread(ys, span * 0.07)
    for (m, ly) in zip(crit.index, label_y):
        on = m in front
        ax.scatter(crit.loc[m, x], crit.loc[m, y], s=110 if on else 70, marker=MARKERS.get(m, "o"),
                   color=COLORS.get(m, INK2), edgecolor=INK if on else "white", linewidth=1.8 if on else 1, zorder=3)
        ax.annotate(METHOD_NAMES.get(m, m) + ("  (Парето)" if on else ""), (crit.loc[m, x], crit.loc[m, y]),
                    xytext=(crit.loc[m, x] * 1.25, ly), textcoords="data", fontsize=8.5, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    ax.set_ylim(min(ys) - span * 0.1, max(label_y) + span * 0.12)
    xs = crit[x].astype(float)
    ax.set_xlim(xs.min() / 1.6, xs.max() * 6)
    ax.set_xscale("log")
    ax.set_xlabel(_label(x) + " (лог. шкала)")
    ax.set_ylabel(_label(y))
    ax.set_title("Аналіз Парето (проєкція на 2 критерії з 14)")
    ax.text(0.01, -0.2, "Обведені маркери – недоміновані за ВСІМА критеріями, не лише за цими двома",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    return _save(fig, path)


def sensitivity_bar(sens: pd.DataFrame, path: Path) -> Path:
    s = sens["Частка перемог, %"].reindex([m for m in COLORS if m in sens.index])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar([METHOD_NAMES[m] for m in s.index], s.values, color=[COLORS[m] for m in s.index],
                  hatch=None, edgecolor="white", linewidth=1.5, width=0.55)
    for b, m in zip(bars, s.index):
        b.set_hatch(HATCH[m])
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f} %", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Частка перемог, %")
    ax.set_title("Аналіз чутливості: 1000 наборів ваг (±30 %)")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def e_vs_weight(curve: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ms = [m for m in COLORS if m in curve.columns]
    for m in ms:
        ax.plot(curve.index, curve[m], color=COLORS[m], linewidth=2, marker=MARKERS[m], markersize=4, markevery=4,
                label=METHOD_NAMES[m])
    ends = _spread([float(curve[m].iloc[-1]) for m in ms], 0.045)
    for m, ly in zip(ms, ends):
        ax.annotate(METHOD_NAMES[m].split()[0], (curve.index[-1], curve[m].iloc[-1]), xytext=(curve.index[-1] + 0.03, ly),
                    textcoords="data", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(curve.index[0] - 0.02, curve.index[-1] + 0.12)
    ax.set_xlabel("Вага критерію К1 «результативність»")
    ax.set_ylabel("Інтегральний показник E")
    ax.set_ylim(0, 1.05)
    ax.set_title("Залежність E від ваги К1")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    return _save(fig, path)
