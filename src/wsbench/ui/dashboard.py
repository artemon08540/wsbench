"""Дашборд для наочного перегляду результатів на захисті (Streamlit).

Запуск:  python -m wsbench dashboard --runs results/main
     або streamlit run src/wsbench/ui/dashboard.py -- --runs results/main
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from wsbench.analytics import stats as S
from wsbench.analytics.metrics import BY_COL, failure_breakdown, load_runs, method_level
from wsbench.decision import mcda
from wsbench.visualization import charts

KEY = ["recall", "precision", "f1", "field_accuracy", "t_ms", "throughput", "traffic_mb", "ram_peak_mb", "cpu_s"]


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/main")
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


@st.cache_data(show_spinner=False)
def _load(path: str, block: str | None) -> pd.DataFrame:
    return load_runs(path, block=block or None)


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.suffix == ".csv":
        return path
    ev = path / "runs_evaluated.csv"
    return ev if ev.exists() else path / "runs.csv"


def main():
    st.set_page_config(page_title="wsbench – результати", layout="wide")
    a = _args()
    st.sidebar.header("Дані")
    runs_in = st.sidebar.text_input("Тека з результатами або runs.csv", a.runs)
    runs_csv = _resolve(runs_in)
    if not runs_csv.exists():
        st.error(f"Не знайдено {runs_csv}")
        st.stop()
    blocks = sorted(_load(str(runs_csv), None).get("block", pd.Series(dtype=str)).dropna().astype(str).unique())
    block = st.sidebar.selectbox("Блок", ["усі"] + blocks, index=0, help="які прогони показувати")
    block = None if block == "усі" else block
    na_tests = st.sidebar.radio("Незастосовні методи у критеріях", ["exclude", "zero"], index=0)
    na_mcda = st.sidebar.radio("Незастосовні методи у виборі методу", ["zero", "exclude"], index=0)
    df = _load(str(runs_csv), block)
    if df.empty:
        st.warning(f"У {runs_csv} немає прогонів для блоку «{block}». Оберіть «усі» в лівій панелі.")
        st.stop()
    metrics = [m for m in KEY if m in df and df[m].notna().any()]
    st.title("Порівняння методів збору даних")
    st.caption(f"{runs_csv} · {len(df)} прогонів · {df['page_id'].nunique()} сторінок · методи: "
               f"{', '.join(sorted(df['method'].unique()))}")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Огляд", "Показники", "Значущість", "Гіпотези", "Вибір методу", "Сирі дані"])

    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Прогонів", len(df))
        c2.metric("Сторінок", df["page_id"].nunique())
        c3.metric("Методів", df["method"].nunique())
        c4.metric("Повторів", int(df["rep"].max()) if "rep" in df and df["rep"].notna().any() else 0)
        mv = method_level(df, na_policy=na_mcda)
        st.subheader("Медіанні значення показників за методами")
        st.dataframe(mv.round(3), use_container_width=True)
        st.subheader("Неуспішні прогони за причинами")
        st.dataframe(failure_breakdown(df), use_container_width=True)

    with t2:
        m = st.selectbox("Показник", metrics, format_func=lambda c: f"{BY_COL[c].code} {BY_COL[c].name}")
        st.dataframe(S.descriptive(df, m), use_container_width=True)
        tmp = Path(tempfile.gettempdir()) / f"wsbench_box_{m}.png"
        charts.boxplot(df, m, tmp, log=m in ("t_ms", "traffic_mb", "cpu_s"))
        st.image(str(tmp), width=760)

    with t3:
        m = st.selectbox("Показник ", metrics, format_func=lambda c: f"{BY_COL[c].code} {BY_COL[c].name}")
        fr = S.friedman(df, m, na_policy=na_tests)
        st.write(f"**Критерій Фрідмана:** χ² = {fr.get('χ²')}, p = {S.fmt_p(fr.get('p'))}, "
                 f"W Кендалла = {fr.get('W Кендалла')}, N = {fr.get('N (сторінок)')}")
        st.dataframe(S.significance_table(df, m, na_policy=na_tests), use_container_width=True)
        st.caption("Суттєво = p (Холм) < 0,05 і |δ| ≥ 0,33 і перевищено поріг практичної значущості.")

    with t4:
        st.subheader("Гіпотеза 1 – повнота за типом рендерингу")
        if "recall" in metrics:
            st.dataframe(pd.DataFrame([S.gap_by_group(df, g1="spa", g2="ssr"), S.gap_by_group(df, g1="hybrid", g2="ssr")]),
                         use_container_width=True)
            tmp = Path(tempfile.gettempdir()) / "wsbench_recall_by_render.png"
            charts.recall_by_render(df, tmp)
            st.image(str(tmp), width=760)
            tmp2 = Path(tempfile.gettempdir()) / "wsbench_scatter.png"
            charts.scatter_recall_time(df, tmp2)
            st.image(str(tmp2), width=760)
        else:
            st.info("Потрібен еталон: python -m wsbench evaluate --runs ...")
        st.subheader("Гіпотеза 2 – у скільки разів М3 «дорожчий» за М1")
        rows = []
        for mm in ["t_ms", "traffic_mb", "ram_peak_mb", "cpu_s"]:
            if mm in metrics:
                t = S.significance_table(df, mm, methods=["M1", "M3"], na_policy=na_tests)
                if len(t):
                    r = t.iloc[0]
                    rows.append({"Показник": BY_COL[mm].name, "М1": r["Медіана A"], "М3": r["Медіана B"],
                                 "М3/М1": r["Медіана B"] / r["Медіана A"] if r["Медіана A"] else None,
                                 "p": r["p (сире)"], "δ": r["δ"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with t5:
        st.subheader("Ваги груп критеріїв")
        c1, c2, c3 = st.columns(3)
        w1 = c1.slider("К1 результативність", 0.0, 1.0, 0.54, 0.01)
        w2 = c2.slider("К2 ресурсомісткість", 0.0, 1.0, 0.16, 0.01)
        w3 = c3.slider("К3 експлуатаційні", 0.0, 1.0, 0.30, 0.01)
        tot = (w1 + w2 + w3) or 1.0
        gw = {"K1": w1 / tot, "K2": w2 / tot, "K3": w3 / tot}
        st.caption(f"Нормовані ваги: К1 = {gw['K1']:.2f}, К2 = {gw['K2']:.2f}, К3 = {gw['K3']:.2f}")
        if "recall" not in metrics:
            st.warning("Немає еталону: повнота (M1–M4) не обчислена, тому оцінка нижче враховує лише швидкість і "
                       "ресурси й завжди віддає перевагу найдешевшому методу. Це НЕ висновок про вибір методу.")
        if df["method"].nunique() < 3:
            st.info("У журналі менше 3 методів – для повного порівняння запустіть усі М1–М4.")
        mv = method_level(df, na_policy=na_mcda)
        crit, const = mcda.drop_constant(mcda.criteria_table(mv))
        _, front = mcda.pareto(crit)
        norm = mcda.normalize(crit)
        e = mcda.weighted_sum(norm, mcda.metric_weights_from_groups(list(norm.columns), gw)).sort_values(ascending=False)
        st.bar_chart(e)
        st.write(f"**Найкращий за цих ваг:** {e.index[0]} (E = {e.iloc[0]:.3f}) · **Множина Парето:** {', '.join(front)}")
        sens = mcda.sensitivity(norm, gw)
        st.dataframe(sens, use_container_width=True)
        with st.expander("Нормалізовані показники (1 = найкраще)"):
            st.dataframe(norm.round(3), use_container_width=True)

    with t6:
        st.dataframe(df, use_container_width=True)


main()
