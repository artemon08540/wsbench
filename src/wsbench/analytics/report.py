"""Повний аналіз журналу прогонів: усі таблиці розділу 3 + графіки + звіт (Markdown і Excel).

Використання з коду:  build_report(runs_csv, out_dir, ahp_path=..., stability_csv=..., sample=...)
З командного рядка:   python -m wsbench analyze --runs results/main
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..decision import mcda
from ..visualization import charts
from . import stats as S
from .effort import effort_table
from .metrics import BY_COL, RUN_METRICS, failure_breakdown, load_runs, method_level

KEY_METRICS = ["recall", "precision", "f1", "field_accuracy", "t_ms", "throughput", "traffic_mb", "ram_peak_mb", "cpu_s"]
H2_METRICS = ["t_ms", "traffic_mb", "ram_peak_mb", "cpu_s"]


# ---------------------------------------------------------------- Markdown без зовнішніх бібліотек
def _fmt(v, col: str = "") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "так" if v else "ні"
    if isinstance(v, (float, np.floating)):
        if col.startswith("p"):
            return S.fmt_p(float(v))
        return f"{v:,.3f}".replace(",", " ").rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    if df is None or len(df) == 0:
        return "_немає даних_\n"
    d = df.reset_index() if index and not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], str(c)) for c in d.columns) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- основна функція
def build_report(runs_csv: Path, out_dir: Path, ahp_path: Optional[Path] = None, stability_csv: Optional[Path] = None,
                 stability_cfg: Optional[Path] = None, sample: Optional[Path] = None, effort_log: Optional[Path] = None,
                 gt_dir: Optional[Path] = None, na_policy_tests: str = "exclude", na_policy_mcda: str = "zero",
                 block: Optional[str] = None) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(runs_csv, block=block)
    has_gt = df["recall"].notna().any() if "recall" in df else False
    metrics = [m for m in KEY_METRICS if m in df and df[m].notna().any()]
    sheets: dict[str, pd.DataFrame] = {}
    md: list[str] = []

    def section(title: str):
        md.append(f"\n## {title}\n")

    md.append(f"# Результати експерименту\n\nЖурнал: `{runs_csv}`  \nСформовано: {datetime.now():%Y-%m-%d %H:%M}  \n")
    if not has_gt:
        md.append("\n> **Увага:** у журналі немає еталону (Recall/Precision/F1 порожні). Показники повноти й "
                  "SR не обчислено; спочатку виконайте `python -m wsbench evaluate --runs ...`.\n")

    # 3.1 характеристика масиву
    section("3.1 Характеристика масиву даних")
    overview = pd.DataFrame({
        "Прогонів": [len(df)], "Сторінок": [df["page_id"].nunique()], "Сайтів": [df["site"].nunique() if "site" in df else np.nan],
        "Методів": [df["method"].nunique()], "Повторів": [df["rep"].max() if "rep" in df else np.nan]})
    md.append(md_table(overview, index=False))
    strata = df.drop_duplicates("page_id").groupby("render_type").size().rename("сторінок").to_frame()
    md.append("\nСтрати за типом рендерингу:\n\n" + md_table(strata))
    fb = failure_breakdown(df)
    sheets["Відмови за причинами"] = fb
    md.append("\nРозклад неуспішних прогонів за причинами (M11):\n\n" + md_table(fb))
    mv = method_level(df, na_policy=na_policy_mcda)
    app = pd.Series(mv.attrs.get("applicability_%", {}), name="Застосовність, %").to_frame()
    md.append("\nЗастосовність методів (частка сторінок, де метод можна було запустити):\n\n" + md_table(app))

    # 3.2 описова статистика
    section("3.2 Описова статистика показників за методами")
    for m in metrics:
        t = S.descriptive(df, m)
        sheets[f"Опис {BY_COL[m].code}"] = t
        md.append(f"\n### {BY_COL[m].code} {BY_COL[m].name}, {BY_COL[m].unit}\n\n" + md_table(t, index=False))
        charts.boxplot(df, m, fig_dir / f"box_{m}.png", log=m in ("t_ms", "traffic_mb", "cpu_s"))
        md.append(f"\n![](figures/box_{m}.png)\n")
    out_list = pd.concat([S.outliers(df, m) for m in metrics], ignore_index=True) if metrics else pd.DataFrame()
    sheets["Викиди (1,5 IQR)"] = out_list
    md.append(f"\nВикидів за правилом 1,5·IQR: **{len(out_list)}** (перелік – аркуш «Викиди» в Excel). "
              "Викиди не видаляються автоматично: кожен перевіряється вручну, рішення – в журналі.\n")

    # 3.3 нормальність
    section("3.3 Перевірка нормальності (Шапіро–Вілк, рівень сторінок)")
    norm = S.normality(df, metrics)
    sheets["Нормальність"] = norm
    share_non = (norm["Нормальний?"] == "ні").mean() * 100 if len(norm) else 0
    md.append(md_table(norm, index=False))
    md.append(f"\nРозподіл відхиляється від нормального у {share_non:.0f} % перевірок → застосовуються "
              "непараметричні критерії.\n")

    # 3.4 Фрідман + значущість
    section("3.4 Порівняння методів: критерій Фрідмана")
    md.append(f"Політика для незастосовних методів у критеріях: **{na_policy_tests}** "
              "(exclude – лише сторінки, де застосовні всі методи; zero – повнота незастосовного методу = 0).\n\n")
    fr = pd.DataFrame([S.friedman(df, m, na_policy=na_policy_tests) for m in metrics])
    sheets["Фрідман"] = fr
    md.append(md_table(fr, index=False))

    section("3.5 Попарні порівняння (Вілкоксон + Холм) – таблиці значущості")
    md.append("Суттєво = p (Холм) < 0,05 **і** |δ| ≥ 0,33 **і** перевищено поріг практичної значущості.\n")
    sig = {m: S.significance_table(df, m, na_policy=na_policy_tests) for m in metrics}
    sig = S.holm_across(sig)
    for m, t in sig.items():
        sheets[f"Значущість {BY_COL[m].code}"] = t
        md.append(f"\n### {BY_COL[m].code} {BY_COL[m].name}\n\n" + md_table(t, index=False))

    # гіпотеза 1
    section("Гіпотеза 1: повнота і тип рендерингу")
    if has_gt:
        h1 = pd.DataFrame([S.gap_by_group(df, g1="spa", g2="ssr"), S.gap_by_group(df, g1="hybrid", g2="ssr"),
                           S.gap_by_group(df, method_hi="M4", g1="spa", g2="ssr")])
        sheets["Г1 SPA vs SSR"] = h1
        md.append(md_table(h1, index=False))
        charts.recall_by_render(df, fig_dir / "recall_by_render.png")
        md.append("\n![](figures/recall_by_render.png)\n")
    else:
        md.append("_потрібен еталон_\n")

    # гіпотеза 2
    section("Гіпотеза 2: ціна повноти (М3 відносно М1)")
    rows = []
    for m in H2_METRICS:
        if m in sig and len(sig[m]):
            r = sig[m][sig[m]["Пара"] == "M1 vs M3"]
            if len(r):
                r = r.iloc[0]
                ratio = r["Медіана B"] / r["Медіана A"] if r["Медіана A"] else np.nan
                rows.append({"Показник": f"{BY_COL[m].code} {BY_COL[m].name}", "Медіана М1": r["Медіана A"],
                             "Медіана М3": r["Медіана B"], "М3/М1, разів": ratio, "≥ 10 разів?": ratio >= 10,
                             "p (Холм)": r["p (Холм)"], "δ": r["δ"], "Ефект": r["Ефект"]})
    h2 = pd.DataFrame(rows)
    sheets["Г2 ціна повноти"] = h2
    md.append(md_table(h2, index=False))
    if metrics:
        charts.scatter_recall_time(df, fig_dir / "scatter_recall_time.png") if has_gt else None
        if has_gt:
            md.append("\n![](figures/scatter_recall_time.png)\n")

    # гіпотеза 3
    section("Гіпотеза 3: стійкість до змін структури (M13)")
    ssr_vals = None
    if stability_csv and Path(stability_csv).exists() and stability_cfg and sample:
        from ..harness.stability import load_stability, ssr_by_method, survival_table
        stab = load_stability(stability_cfg, sample)
        surv = survival_table(Path(stability_csv), stab, gt_dir or Path("ground_truth"))
        ssr_t = ssr_by_method(surv)
        ssr_vals = ssr_t["SSR, %"].to_dict()
        tests = S.survival_tests(surv)
        sheets["Г3 виживання (деталі)"] = surv
        sheets["Г3 SSR"] = ssr_t
        sheets["Г3 Фішер"] = tests
        md.append(md_table(ssr_t) + "\n" + md_table(tests, index=False))
        if tests.attrs.get("chi2"):
            c = tests.attrs["chi2"]
            md.append(f"\nχ² по всіх методах: χ² = {c['χ²']}, df = {c['df']}, p = {S.fmt_p(c['p'])}\n")
        ref_bad = surv[surv["is_reference"] & (surv["works"] == False)]  # noqa: E712
        if len(ref_bad):
            md.append(f"\n> Увага: {len(ref_bad)} збирачів не працюють навіть на еталонній (найстарішій) версії – "
                      "їх треба доналаштувати, інакше M13 занижено.\n")
    else:
        md.append("_блок стійкості не виконано (див. `python -m wsbench stability-run`)_\n")

    # кореляції
    section("3.7 Кореляційний аналіз (Спірмен)")
    corr = S.correlations(df)
    sheets["Кореляції"] = corr
    md.append(md_table(corr, index=False))
    red = S.metric_redundancy(df, metrics)
    sheets["Надлишковість |ρ|>0,9"] = red
    md.append("\nПари показників з |ρ| > 0,9 (кандидати на надлишковість):\n\n" + md_table(red, index=False))
    if len(metrics) >= 3:
        charts.corr_heatmap(df, metrics, fig_dir / "corr_heatmap.png")
        md.append("\n![](figures/corr_heatmap.png)\n")

    # трудомісткість
    extra = {}
    if ssr_vals:
        extra["ssr"] = ssr_vals
    if sample and Path(sample).exists():
        eff = effort_table(Path(sample), Path(effort_log or "journal/effort_log.csv"), list(df["page_id"].unique()))
        sheets["M14 трудомісткість"] = eff
        section("M14 Трудомісткість підготовки")
        md.append(md_table(eff))
        extra["effort_loc"] = eff["LOC (медіана)"].dropna().to_dict()

    # багатокритеріальне оцінювання
    section("3.8 Багатокритеріальне оцінювання")
    mv = method_level(df, extra=extra, na_policy=na_policy_mcda)
    ahp_res = mcda.load_ahp(ahp_path) if ahp_path and Path(ahp_path).exists() else None
    r = mcda.evaluate_all(mv, ahp_res)
    sheets["БКО значення показників"] = r["criteria"]
    sheets["БКО нормалізовані"] = r["normalized"]
    sheets["БКО домінування"] = r["dominance"]
    sheets["БКО E за сценаріями"] = r["E"]
    sheets["БКО ранги"] = r["ranks"]
    sheets["БКО чутливість"] = r["sensitivity"]
    sheets["БКО E від ваги К1"] = r["e_vs_k1"]
    md.append(f"Політика незастосовності: **{na_policy_mcda}**.\n\n### 3.8.1 Аналіз Парето\n\nЗначення показників:\n\n"
              + md_table(r["criteria"]))
    if r["constant_dropped"]:
        md.append(f"\nНе враховано (однакові для всіх методів, не розрізняють їх): {', '.join(r['constant_dropped'])}\n")
    md.append("\nМатриця домінування (рядок домінує над стовпцем):\n\n" + md_table(r["dominance"]))
    md.append(f"\n**Множина Парето:** {', '.join(r['pareto_front'])}\n")
    charts.pareto_chart(r["criteria"], r["pareto_front"], fig_dir / "pareto.png")
    md.append("\n![](figures/pareto.png)\n")
    if ahp_res:
        md.append("\n### Ваги AHP (Сааті)\n\n" + md_table(pd.Series(ahp_res.weights, name="вага").to_frame()) +
                  f"\nλmax = {ahp_res.lambda_max:.4f}; CI = {ahp_res.ci:.4f}; CR = {ahp_res.cr:.4f} → "
                  f"{'узгоджено (CR < 0,1)' if ahp_res.consistent else '**НЕ узгоджено (CR ≥ 0,1) – перегляньте судження**'}\n")
    md.append("\n### 3.8.2 Інтегральний показник E (безрозмірний, 0–1)\n\n" + md_table(r["E"]) +
              "\nРанги:\n\n" + md_table(r["ranks"]))
    md.append("\nВаги сценаріїв (К1 / К2 / К3):\n\n" + md_table(pd.DataFrame(r["scenarios"]).T))
    md.append("\n### 3.8.3 Аналіз чутливості\n\nЗбурення ваг окремих показників ±30 %:\n\n" + md_table(r["sensitivity"]) +
              "\nЗбурення ваг груп К1–К3 ±30 %:\n\n" + md_table(r["sensitivity_groups"]))
    charts.sensitivity_bar(r["sensitivity"], fig_dir / "sensitivity.png")
    charts.e_vs_weight(r["e_vs_k1"], fig_dir / "e_vs_k1.png")
    md.append("\n![](figures/sensitivity.png)\n\n![](figures/e_vs_k1.png)\n")

    # збереження
    report_md = out_dir / "report.md"
    report_md.write_text("\n".join(md), encoding="utf-8")
    with pd.ExcelWriter(out_dir / "tables.xlsx") as xw:
        for name, t in sheets.items():
            if t is None or len(t) == 0:
                continue
            safe = name.replace("/", "-")[:31]
            t.to_excel(xw, sheet_name=safe)
    (out_dir / "mcda.json").write_text(json.dumps({
        "pareto_front": r["pareto_front"], "E": r["E"].round(4).to_dict(), "scenarios": r["scenarios"],
        "ahp": ahp_res.__dict__ if ahp_res else None, "na_policy_mcda": na_policy_mcda}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    return report_md
