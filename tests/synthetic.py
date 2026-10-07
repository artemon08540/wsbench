"""Генератор СИНТЕТИЧНОГО журналу прогонів (40 сторінок × 4 методи × 5 повторів = 800 рядків).

Потрібен лише для перевірки аналітичного конвеєра до появи реальних даних.
Ці числа НЕ можна використовувати в роботі. Запуск:
    python tests/synthetic.py results/synthetic/runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(path: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    pages = []
    for i in range(40):
        rt = ["ssr"] * 14 + ["spa"] * 13 + ["hybrid"] * 13
        pages.append({"page_id": f"p{i:02d}", "site": f"site{i // 2}", "render_type": rt[i],
                      "pagination": ["classic", "load_more", "infinite"][i % 3], "n_true": int(rng.integers(15, 61)),
                      "has_api": rng.random() < 0.6, "blocks_http": rng.random() < 0.08})
    rows = []
    for p in pages:
        share_m1 = {"ssr": 1.0, "hybrid": rng.uniform(0.2, 0.8), "spa": rng.uniform(0, 0.15)}[p["render_type"]]
        for rep in range(1, 6):
            for m in ["M1", "M2", "M3", "M4"]:
                n = p["n_true"]
                r = {"run_id": f"{p['page_id']}__{m}__r{rep}", "block": "main", "page_id": p["page_id"], "site": p["site"],
                     "render_type": p["render_type"], "pagination": p["pagination"], "method": m, "rep": rep,
                     "error_type": "", "antibot_hits": 0, "status_code": 200}
                if m == "M2" and not p["has_api"]:
                    r.update(error_type="not_applicable", n_found=0, success=False)
                    rows.append(r)
                    continue
                m1_t = rng.lognormal(np.log(600), 0.25)
                m3_t = rng.lognormal(np.log(3500 + 40 * n), 0.15)
                if m == "M1":
                    if p["blocks_http"]:
                        r.update(error_type="http_403", antibot_hits=1, status_code=403, n_found=0, tp=0, fp=0, fn=n,
                                 t_ms=rng.lognormal(np.log(250), 0.2), traffic_mb=0.01, ram_peak_mb=rng.normal(122, 4),
                                 cpu_s=rng.normal(0.2, 0.03))
                    else:
                        tp = int(round(n * share_m1))
                        r.update(tp=tp, fp=int(rng.random() < 0.2), fn=n - tp, t_ms=m1_t,
                                 traffic_mb=rng.normal(0.15, 0.03), ram_peak_mb=rng.normal(126, 6), cpu_s=rng.normal(0.35, 0.05),
                                 html_size=int(rng.normal(400_000, 120_000)))
                elif m == "M2":
                    tp = n - int(rng.random() < 0.3)
                    r.update(tp=tp, fp=0, fn=n - tp, t_ms=rng.lognormal(np.log(420), 0.25), traffic_mb=rng.normal(0.06, 0.01),
                             ram_peak_mb=rng.normal(120, 5), cpu_s=rng.normal(0.25, 0.04))
                elif m == "M3":
                    tp = n - int(rng.random() < 0.15)
                    r.update(tp=tp, fp=int(rng.random() < 0.3), fn=n - tp, t_ms=m3_t, traffic_mb=rng.normal(2.5, 0.5),
                             ram_peak_mb=rng.normal(620, 60), cpu_s=rng.normal(3.6, 0.6), n_xhr=int(rng.integers(5, 60)),
                             n_requests=int(rng.integers(60, 250)))
                    r["t_ms"] += 30 * r["n_xhr"]
                else:  # M4
                    fallback = p["blocks_http"] or share_m1 < 0.8
                    tp = (n - int(rng.random() < 0.15)) if fallback else n
                    r.update(tp=tp, fp=0, fn=n - tp, t_ms=(m1_t + m3_t) if fallback else m1_t,
                             traffic_mb=(2.6 if fallback else 0.15) + rng.normal(0, 0.05),
                             ram_peak_mb=rng.normal(625 if fallback else 127, 20), cpu_s=rng.normal(3.8 if fallback else 0.36, 0.1),
                             fallback=fallback)
                if r.get("error_type") == "" and rng.random() < 0.01:
                    r.update(error_type="timeout", t_ms=60_500)
                tp, fp, fn = r.get("tp", 0), r.get("fp", 0), r.get("fn", n)
                r["n_found"] = tp + fp
                r["recall"] = 100 * tp / (tp + fn) if tp + fn else 0
                r["precision"] = 100 * tp / (tp + fp) if tp + fp else 0
                r["f1"] = (2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"])) if r["recall"] + r["precision"] else 0
                r["field_accuracy"] = 100 * rng.uniform(0.93, 1.0) if tp else np.nan
                r["throughput"] = tp / (r["t_ms"] / 1000)
                if r["error_type"] == "" and r["n_found"] == 0:
                    r["error_type"] = "empty"
                r["success"] = (r["error_type"] in ("", "empty")) and r["recall"] >= 90
                rows.append(r)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/synthetic/runs.csv")
    print("Синтетичний журнал:", make(out))
