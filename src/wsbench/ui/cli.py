"""Інтерфейс командного рядка wsbench.

Приклади:
  python -m wsbench demo
  python -m wsbench check-site --sample config/sites.yaml
  python -m wsbench pilot --sample config/sites.yaml --methods M1,M3 --repeats 3
  python -m wsbench run-experiment --sample config/sites.yaml --repeats 5 --seed 42 --gt ground_truth
  python -m wsbench evaluate --runs results/main --gt ground_truth
  python -m wsbench gt-draft --sample config/sites.yaml --page rozetka_laptops
  python -m wsbench env-info
  python -m wsbench overhead-test --sample config/sites.yaml --page demo_ssr
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

from ..collectors import build_collectors
from ..groundtruth import evaluate, load_ground_truth, write_ground_truth_template
from ..harness import RunExecutor, load_sample, plan_runs
from ..domain import ProductRecord
from . import report

PROJECT_ROOT = Path(__file__).resolve().parents[3]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def _print_row(row: dict) -> None:
    rec = row.get("recall")
    rec_s = f" recall={rec:>5}%" if rec not in (None, "") else ""
    print(f"[{row['order_idx']:>4}] {row['page_id']:<24} {row['method']} r{row['rep']}  "
          f"found={row.get('n_found', 0):>4}{rec_s}  t={row.get('t_ms', '-'):>9} ms  "
          f"RAM={row.get('ram_peak_mb', '-'):>8} MB  {row.get('error_type') or 'ok'}", flush=True)


def _select_pages(sample: Path, page_ids: str | None):
    pages = load_sample(sample)
    if page_ids:
        wanted = set(page_ids.split(","))
        pages = [p for p in pages if p.page_id in wanted]
        if not pages:
            sys.exit(f"Сторінок {wanted} у {sample} не знайдено")
    return pages


def _run(args, block: str) -> Path:
    pages = _select_pages(args.sample, args.pages)
    methods = args.methods.split(",")
    tasks = plan_runs(pages, methods, args.repeats, args.seed)
    collectors = build_collectors(methods, hybrid_threshold=args.threshold, headless=not args.headed)
    out = Path(args.out)
    print(f"План: {len(pages)} сторінок × {len(methods)} методів × {args.repeats} повторів = {len(tasks)} прогонів; "
          f"зерно={args.seed}; журнал: {out / 'runs.csv'}")
    ex = RunExecutor(collectors, out, block=block, gt_dir=args.gt, min_delay_s=args.min_delay,
                     timeout_s=args.timeout, on_row=_print_row)
    log = ex.run(tasks)
    df = report.load_runs(log)
    table = report.pilot_table(df)
    table.to_csv(out / "summary_by_page.csv", index=False, encoding="utf-8-sig")
    print("\nМедіани за повторами (сторінка × метод):")
    print(table.to_string(index=False))
    print("\nМедіани за методами:")
    print(report.method_table(df).to_string())
    return log


def cmd_pilot(args):
    _run(args, block="pilot")


def cmd_run(args):
    _run(args, block="main")


def cmd_check(args):
    from .inspect_site import inspect_page
    pages = _select_pages(args.sample, args.pages)
    out = Path(args.out)
    checks_path = out / "checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}
    for p in pages:
        print(f"\n=== {p.page_id}  {p.url}")
        c = inspect_page(p, headless=not args.headed, debug_dir=out / "debug" / p.page_id)
        d = c.__dict__
        checks[p.page_id] = d
        print(f"robots.txt дозволяє: {c.robots_allowed}   HTTP: {c.raw_status}   анти-бот: {c.antibot or '-'}")
        shells = f" (порожніх заготовок: {c.raw_shells - c.raw_cards})" if c.raw_shells > c.raw_cards else ""
        print(f"заповнених карток у сирому HTML: {c.raw_cards}{shells}   у DOM після рендерингу: {c.rendered_cards}   "
              f"частка: {c.raw_share if c.raw_share is None else round(c.raw_share * 100, 1)}%   -> тип: {c.render_type_auto}")
        print(f"перші товари: {c.first_names}")
        if c.final_url and c.final_url.split("#")[0] != p.url.split("#")[0]:
            print(f"   браузер опинився на іншій адресі: {c.final_url[:150]}")
        if c.rendered_cards == 0:
            print(f"   !! браузер не знайшов жодної картки – подивіться {out / 'debug' / p.page_id / 'screenshot.png'}")
        if c.render_type_auto == "blocked":
            print("   !! сайт заблокував HTTP-запит Python (М1/М2). Тип рендерингу так визначити не можна – "
                  "це результат для показника M12, а не ознака SPA.")
        if c.embedded_data:
            print(f"вбудовані дані в HTML: {c.embedded_data}")
        cand = [j for j in c.json_candidates if j['contains_first_product']]
        print(f"JSON-відповідей (XHR/fetch): {len(c.json_candidates)}; з назвою першого товару: {len(cand)}")
        for j in cand[:5]:
            print(f"   М2-кандидат: {j['method']} {j['url'][:160]}  ({j['bytes']} B)")
        if not cand:
            for j in c.json_candidates[:3]:
                print(f"   перевірте вручну: {j['method']} {j['url'][:160]}  ({j['bytes']} B)")
        report.save_json(checks, checks_path)
    report.export_sample_xlsx(load_sample(args.sample), checks, out / "sample.xlsx")
    print(f"\nЗбережено: {checks_path} і {out / 'sample.xlsx'}")


def cmd_evaluate(args):
    """Перерахунок M1–M4 із «сирих» JSON без повторного збору (ФВ-5, ФВ-6)."""
    run_dir = Path(args.runs)
    runs = pd.read_csv(run_dir / "runs.csv")
    upd = []
    for _, r in runs.iterrows():
        raw_p = run_dir / "raw" / f"{r['run_id']}.json"
        truth = load_ground_truth(Path(args.gt), r["page_id"])
        if truth is None or not raw_p.exists():
            upd.append({})
            continue
        raw = json.loads(raw_p.read_text(encoding="utf-8"))
        recs = [ProductRecord(**x) for x in raw["result"]["records"]]
        q = evaluate(recs, truth).as_dict()
        t = float(r["t_ms"]) if pd.notna(r.get("t_ms")) else None
        q["throughput"] = round(q["tp"] / (t / 1000), 3) if t else None
        tech_ok = str(r.get("error_type") or "") in ("", "nan", "empty") and (pd.isna(r.get("status_code")) or r["status_code"] < 400)
        q["success"] = bool(tech_ok and q["recall"] >= 90)
        upd.append(q)
    upd_df = pd.DataFrame(upd, index=runs.index)
    for c in upd_df.columns:
        runs[c] = upd_df[c].combine_first(runs[c]) if c in runs else upd_df[c]
    out = run_dir / "runs_evaluated.csv"
    runs.to_csv(out, index=False, encoding="utf-8")
    print(report.method_table(report.load_runs(out)).to_string())
    print(f"\nЗбережено: {out}")


def cmd_gt_draft(args):
    pages = _select_pages(args.sample, args.pages)
    coll = build_collectors([args.method], headless=not args.headed)[args.method]
    for p in pages:
        res = coll.collect(p, 60)
        path = Path(args.gt) / f"{p.page_id}.draft.csv"
        write_ground_truth_template(path, res.records)
        print(f"{p.page_id}: {len(res.records)} записів -> {path}")
    print("\nУВАГА: це лише чернетка. Відкрийте її в Excel, звірте КОЖЕН рядок зі сторінкою в браузері (п. 7.2 ТЗ),\n"
          "допишіть пропущені товари, видаліть зайві, виправте ціни. Збережіть як «CSV (роздільник – крапка з комою)»\n"
          "і перейменуйте файл на <page_id>.csv (без .draft).")


def cmd_gt_agreement(args):
    """Надійність еталону: збіг першої розмітки (ground_truth/<id>.csv) з повторною (ground_truth/second_pass/<id>.csv)."""
    from ..groundtruth import agreement, read_gt_file
    gt, second = Path(args.gt), Path(args.gt) / "second_pass"
    ids = args.pages.split(",") if args.pages else sorted(p.stem for p in second.glob("*.csv"))
    if not ids:
        sys.exit(f"У {second} немає файлів повторної розмітки")
    rows, tot_same, tot_union = [], 0, 0
    for pid in ids:
        a, b = gt / f"{pid}.csv", second / f"{pid}.csv"
        if not a.exists() or not b.exists():
            print(f"{pid}: немає {a if not a.exists() else b}")
            continue
        r = agreement(read_gt_file(a), read_gt_file(b))
        union = r["записів_1"] + r["записів_2"] - r["спільних назв"]
        tot_same += r["збіг назва+ціна"]
        tot_union += union
        rows.append(r)
        print(f"\n{pid}: збіг {r['збіг, %']} %  (1-ша: {r['записів_1']}, 2-га: {r['записів_2']}, однакових: {r['збіг назва+ціна']})")
        for n in r["лише в 1-й"]:
            print(f"   лише в 1-й розмітці: {n}")
        for n in r["лише в 2-й"]:
            print(f"   лише в 2-й розмітці: {n}")
        for n, p1, p2 in r["різні ціни"]:
            print(f"   різні ціни: {n}: {p1} vs {p2}")
    if tot_union:
        total = 100 * tot_same / tot_union
        print(f"\nЗАГАЛОМ: збіг {total:.2f} % → {'розмітка надійна (≥ 98 %)' if total >= 98 else 'НИЖЧЕ 98 % – розберіть розбіжності'}")


def cmd_env(args):
    info = report.environment_info()
    report.save_json(info, Path(args.out) / "environment.json")
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_overhead(args):
    """НФВ-6: накладні витрати монітора. N прогонів М1 з монітором і без, порівняння медіан."""
    import statistics
    import time
    from ..harness import ResourceMonitor
    page = _select_pages(args.sample, args.pages)[0]
    coll = build_collectors(["M1"])["M1"]
    times = {True: [], False: []}
    order = [True, False] * args.n
    import random
    random.Random(args.seed).shuffle(order)
    for flag in order:
        mon = ResourceMonitor(enabled=flag)
        mon.start()
        t0 = time.perf_counter()
        coll.collect(page, 60)
        times[flag].append((time.perf_counter() - t0) * 1000)
        mon.stop()
        time.sleep(args.min_delay)
    on, off = statistics.median(times[True]), statistics.median(times[False])
    print(f"медіана з монітором: {on:.2f} мс; без: {off:.2f} мс; різниця: {(on - off) / off * 100:+.2f} %  (n={args.n} кожен)")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass


def _serve_demo(port: int) -> ThreadingHTTPServer:
    handler = partial(_QuietHandler, directory=str(PROJECT_ROOT / "demo" / "site"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def cmd_demo(args):
    if not (PROJECT_ROOT / "demo" / "site" / "ssr.html").exists():
        import runpy
        runpy.run_path(str(PROJECT_ROOT / "demo" / "build_demo.py"), run_name="__main__")
    srv = _serve_demo(8765)
    try:
        ns = argparse.Namespace(sample=PROJECT_ROOT / "config" / "demo_sites.yaml", pages=None,
                                methods=args.methods, repeats=args.repeats, seed=42, threshold=0.8,
                                headed=False, out=args.out, gt=PROJECT_ROOT / "demo" / "ground_truth",
                                min_delay=0.0, timeout=60)
        _run(ns, block="demo")
    finally:
        srv.shutdown()



def cmd_analyze(args):
    """Розділ 3 цілком: статистика, гіпотези, багатокритеріальне оцінювання, графіки, звіт."""
    from ..analytics.report import build_report
    runs = Path(args.runs)
    runs_csv = runs / "runs_evaluated.csv" if (runs / "runs_evaluated.csv").exists() else runs / "runs.csv"
    if runs.suffix == ".csv":
        runs_csv = runs
    out = Path(args.out) if args.out else runs_csv.parent / "report"
    stab = Path(args.stability) if args.stability else None
    path = build_report(runs_csv, out, ahp_path=Path(args.ahp), stability_csv=stab,
                        stability_cfg=Path(args.stability_cfg), sample=Path(args.sample), effort_log=Path(args.effort_log),
                        gt_dir=Path(args.gt), na_policy_tests=args.na_tests, na_policy_mcda=args.na_mcda, block=args.block)
    print(f"Журнал: {runs_csv}")
    print(f"Звіт:   {path}\nТаблиці: {out / 'tables.xlsx'}\nГрафіки: {out / 'figures'}")


def cmd_decide(args):
    """Лише багатокритеріальне оцінювання (швидко подивитися E при інших вагах)."""
    from ..analytics.metrics import load_runs, method_level
    from ..decision import mcda
    runs = Path(args.runs)
    runs_csv = runs if runs.suffix == ".csv" else (runs / "runs_evaluated.csv" if (runs / "runs_evaluated.csv").exists() else runs / "runs.csv")
    mv = method_level(load_runs(runs_csv), na_policy=args.na_mcda)
    a = mcda.load_ahp(Path(args.weights)) if Path(args.weights).exists() else None
    if a:
        print(f"AHP: ваги {{{', '.join(f'{k}: {v:.3f}' for k, v in a.weights.items())}}}; CR = {a.cr:.4f} "
              f"({'узгоджено' if a.consistent else 'НЕ узгоджено – перегляньте матрицю'})")
    r = mcda.evaluate_all(mv, a)
    print("\nМножина Парето:", ", ".join(r["pareto_front"]))
    print("\nІнтегральний показник E:\n", r["E"].round(3).to_string())
    print("\nЧутливість (±30 %):\n", r["sensitivity"].to_string())


def cmd_stability_find(args):
    """Знайти в Wayback Machine копії сторінок ~24 і ~12 міс. тому і підготувати config/stability.yaml."""
    import yaml as _yaml
    from ..harness.stability import suggest_snapshots
    pages = _select_pages(args.sample, args.pages)
    sugg = suggest_snapshots(pages)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} вже існує – знайдені копії лише показую (додайте --force, щоб перезаписати):")
    for s_ in sugg:
        print(f"  {s_['page_id']}: {s_['snapshots']}")
    if not out.exists() or args.force:
        doc = {"pages": [{"page_id": s_["page_id"], "snapshots": s_["snapshots"],
                          "expected_counts": {str(t): None for t in s_["snapshots"]}} for s_ in sugg]}
        out.parent.mkdir(parents=True, exist_ok=True)
        header = ("# Блок стійкості (п. 7.4 ТЗ). Для кожної сторінки: версії від найстарішої до live.\n"
                  "# 1) Відкрийте найстарішу копію (https://web.archive.org/web/<дата>/<url>) і, якщо треба, допишіть сюди\n"
                  "#    card_selector / name_selector / price_selector / api, налаштовані САМЕ НА НЕЇ.\n"
                  "# 2) expected_counts – скільки товарів на кожній версії (порахувати вручну), якщо немає еталону\n"
                  "#    ground_truth/<page_id>@<дата>.csv. Сторінки з «НЕ ЗНАЙДЕНО» видаліть і запишіть у журнал.\n")
        out.write_text(header + _yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"\nЗбережено шаблон: {out}")


def cmd_stability_run(args):
    from ..harness.stability import load_stability, run_stability, ssr_by_method, survival_table
    stab = load_stability(Path(args.config), Path(args.sample))
    methods = args.methods.split(",")
    collectors = build_collectors(methods, hybrid_threshold=args.threshold, headless=not args.headed)
    out = Path(args.out)
    n = sum(len(s_.snapshots) for s_ in stab) * len(methods)
    print(f"Блок стійкості: {len(stab)} сторінок × версії × {len(methods)} методів = {n} прогонів; журнал: {out / 'runs.csv'}")
    log = run_stability(stab, collectors, out, Path(args.gt), min_delay_s=args.min_delay, on_row=_print_row)
    surv = survival_table(log, stab, Path(args.gt))
    surv.to_csv(out / "survival.csv", index=False, encoding="utf-8-sig")
    print("\nM13 – частка збирачів, що вижили на новіших версіях:\n", ssr_by_method(surv).to_string())


def cmd_effort(args):
    from ..analytics.effort import effort_table
    print(effort_table(Path(args.sample), Path(args.log)).to_string())


def cmd_dashboard(args):
    import subprocess
    app = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--", "--runs", str(args.runs)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsbench", description="Порівняльне дослідження методів збору даних")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, methods="M1,M2,M3,M4", repeats=5, out="results/main"):
        p.add_argument("--sample", type=Path, default=Path("config/sites.yaml"), help="файл вибірки (YAML)")
        p.add_argument("--pages", help="лише ці page_id через кому")
        p.add_argument("--methods", default=methods)
        p.add_argument("--repeats", type=int, default=repeats)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--threshold", type=float, default=0.8, help="поріг М4 (частка від expected_count)")
        p.add_argument("--gt", type=Path, default=Path("ground_truth"), help="тека з еталоном")
        p.add_argument("--out", default=out)
        p.add_argument("--min-delay", type=float, default=2.0, help="мін. пауза між зверненнями до домену, с")
        p.add_argument("--timeout", type=int, default=60)
        p.add_argument("--headed", action="store_true", help="показувати вікно браузера (для налагодження)")

    p = sub.add_parser("pilot", help="пілотний збір (розділ 13 ТЗ)")
    common(p, methods="M1,M3", repeats=3, out="results/pilot")
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("run-experiment", help="основний експеримент")
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check-site", help="robots.txt, тип рендерингу, пошук endpoint для М2")
    p.add_argument("--sample", type=Path, default=Path("config/sites.yaml"))
    p.add_argument("--pages")
    p.add_argument("--out", default="results/checks")
    p.add_argument("--headed", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("evaluate", help="перерахувати Recall/Precision/F1/FA із сирих результатів")
    p.add_argument("--runs", required=True)
    p.add_argument("--gt", default="ground_truth")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("gt-draft", help="чернетка еталону для ручної перевірки")
    p.add_argument("--sample", type=Path, default=Path("config/sites.yaml"))
    p.add_argument("--pages")
    p.add_argument("--method", default="M3")
    p.add_argument("--gt", default="ground_truth")
    p.add_argument("--headed", action="store_true")
    p.set_defaults(func=cmd_gt_draft)

    p = sub.add_parser("gt-agreement", help="надійність еталону: збіг з повторною розміткою (≥ 98 %)")
    p.add_argument("--gt", default="ground_truth")
    p.add_argument("--pages")
    p.set_defaults(func=cmd_gt_agreement)

    p = sub.add_parser("env-info", help="зафіксувати умови проведення (ОС, залізо, версії)")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("overhead-test", help="накладні витрати монітора (НФВ-6)")
    p.add_argument("--sample", type=Path, default=Path("config/sites.yaml"))
    p.add_argument("--pages")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-delay", type=float, default=2.0)
    p.set_defaults(func=cmd_overhead)


    p = sub.add_parser("analyze", help="статистика, гіпотези, багатокритеріальне оцінювання, графіки, звіт")
    p.add_argument("--runs", required=True, help="тека з runs.csv (або сам файл .csv)")
    p.add_argument("--out", help="куди зберегти звіт (за замовчуванням <runs>/report)")
    p.add_argument("--ahp", default="config/ahp.json")
    p.add_argument("--sample", default="config/sites.yaml")
    p.add_argument("--gt", default="ground_truth")
    p.add_argument("--effort-log", default="journal/effort_log.csv")
    p.add_argument("--stability", help="журнал блоку стійкості (results/stability/runs.csv)")
    p.add_argument("--stability-cfg", default="config/stability.yaml")
    p.add_argument("--na-tests", choices=["exclude", "zero"], default="exclude",
                   help="незастосовні методи у критеріях: exclude – лише повні блоки; zero – повнота = 0")
    p.add_argument("--na-mcda", choices=["zero", "exclude"], default="zero",
                   help="незастосовні методи в багатокритеріальному оцінюванні (за замовч. zero)")
    p.add_argument("--block", help="аналізувати лише цей блок (main / pilot)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("decide", help="лише багатокритеріальне оцінювання")
    p.add_argument("--runs", required=True)
    p.add_argument("--weights", default="config/ahp.json")
    p.add_argument("--na-mcda", choices=["zero", "exclude"], default="zero")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("stability-find", help="знайти архівні копії у Wayback Machine (крок 1 блоку стійкості)")
    p.add_argument("--sample", type=Path, default=Path("config/sites.yaml"))
    p.add_argument("--pages")
    p.add_argument("--out", default="config/stability.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_stability_find)

    p = sub.add_parser("stability-run", help="прогони блоку стійкості (M13)")
    p.add_argument("--config", default="config/stability.yaml")
    p.add_argument("--sample", default="config/sites.yaml")
    p.add_argument("--methods", default="M1,M2,M3,M4")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--gt", default="ground_truth")
    p.add_argument("--out", default="results/stability")
    p.add_argument("--min-delay", type=float, default=4.0, help="пауза між зверненнями до web.archive.org, с")
    p.add_argument("--headed", action="store_true")
    p.set_defaults(func=cmd_stability_run)

    p = sub.add_parser("effort", help="M14: трудомісткість (LOC з sites.yaml + хвилини з журналу)")
    p.add_argument("--sample", default="config/sites.yaml")
    p.add_argument("--log", default="journal/effort_log.csv")
    p.set_defaults(func=cmd_effort)

    p = sub.add_parser("dashboard", help="дашборд Streamlit для перегляду результатів")
    p.add_argument("--runs", default="results/main")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("demo", help="прогін на локальному демо-сайті (перевірка, що все встановлено)")
    p.add_argument("--methods", default="M1,M2,M3,M4")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", default="results/demo")
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
