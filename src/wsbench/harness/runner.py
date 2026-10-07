"""Раннер експерименту: план прогонів, рандомізація, ізоляція, вимірювання, журнал.

Протокол одного прогону (п. 7.3 ТЗ):
  1) взяти завдання (сторінка P, метод M, повтор k) з перемішаної черги;
  2) пауза – не менше min_delay_s від попереднього звернення до того самого домену;
  3) чистий стан: кожен збирач сам створює новий HTTP-клієнт / новий браузер і контекст;
  4) зафіксувати початковий стан ресурсів, увімкнути монітор;
  5) запустити таймер perf_counter();
  6) виконати збір методом M;
  7) зупинити таймер і монітор;
  8) зберегти «сирий» результат у JSON;
  9) дописати рядок у журнал (CSV, flush після кожного прогону – відмовостійкість);
 10) перейти до наступного завдання.
Повторний запуск з тим самим журналом пропускає вже виконані прогони (відновлення після збою).
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import yaml

from ..domain import ICollector, PageSpec
from ..groundtruth import evaluate, load_ground_truth
from .monitor import ResourceMonitor

SUCCESS_RECALL = 90.0

LOG_FIELDS = [
    "run_id", "block", "order_idx", "started_at", "page_id", "site", "render_type", "pagination",
    "method", "rep",
    # якість (M1–M4) і надійність (M5, M11, M12)
    "n_found", "tp", "fp", "fn", "recall", "precision", "f1", "field_accuracy", "n_fuzzy",
    "success", "status_code", "error_type", "error_msg", "antibot_hits",
    # продуктивність і ресурси (M6–M10)
    "t_ms", "throughput", "traffic_mb", "body_mb", "ram_peak_mb", "ram_baseline_mb", "cpu_s",
    # характеристики сторінки / службові
    "n_requests", "n_xhr", "html_size", "fallback", "m1_found", "browser_launch_ms", "monitor_samples",
]


def load_sample(path: Path) -> list[PageSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pages = data["pages"] if isinstance(data, dict) else data
    return [PageSpec.from_dict(p) for p in pages]


@dataclass
class Task:
    page: PageSpec
    method: str
    rep: int

    @property
    def run_id(self) -> str:
        return f"{self.page.page_id}__{self.method}__r{self.rep}"


def plan_runs(pages: list[PageSpec], methods: list[str], repeats: int, seed: int) -> list[Task]:
    """Повний факторний план P × M × k, перемішаний ОДИН раз із фіксованим зерном."""
    tasks = [Task(p, m, k) for p in pages for m in methods for k in range(1, repeats + 1)]
    random.Random(seed).shuffle(tasks)
    return tasks


class RunExecutor:
    def __init__(
        self,
        collectors: dict[str, ICollector],
        out_dir: Path,
        block: str = "main",
        gt_dir: Optional[Path] = None,
        min_delay_s: float = 2.0,
        timeout_s: int = 60,
        monitor: bool = True,
        on_row: Optional[Callable[[dict], None]] = None,
    ):
        self.collectors = collectors
        self.out_dir = Path(out_dir)
        self.raw_dir = self.out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "runs.csv"
        self.block = block
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.min_delay_s = min_delay_s
        self.timeout_s = timeout_s
        self.monitor_enabled = monitor
        self.on_row = on_row
        self._last_hit: dict[str, float] = {}

    # --- службове -----------------------------------------------------------------------
    def _done_ids(self) -> set[str]:
        if not self.log_path.exists():
            return set()
        with self.log_path.open(encoding="utf-8", newline="") as f:
            return {r["run_id"] for r in csv.DictReader(f)}

    def _append(self, row: dict) -> None:
        new = not self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()

    def _polite_wait(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.min_delay_s - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)

    def _mark_hit(self, url: str) -> None:
        self._last_hit[urlparse(url).netloc] = time.monotonic()

    # --- основний цикл --------------------------------------------------------------------
    def run(self, tasks: list[Task]) -> Path:
        done = self._done_ids()
        for idx, task in enumerate(tasks, start=1):
            if task.run_id in done:
                continue
            row = self.run_one(task, idx)
            self._append(row)
            if self.on_row:
                self.on_row(row)
        return self.log_path

    def run_one(self, task: Task, order_idx: int) -> dict:
        page, collector = task.page, self.collectors[task.method]
        row = {
            "run_id": task.run_id, "block": self.block, "order_idx": order_idx,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "page_id": page.page_id, "site": page.site, "render_type": page.render_type,
            "pagination": page.pagination, "method": task.method, "rep": task.rep,
        }
        if not collector.is_applicable(page):
            row.update(error_type="not_applicable", n_found=0, success=False)
            return row

        self._polite_wait(page.url)
        mon = ResourceMonitor(enabled=self.monitor_enabled)
        mon.start()
        t0 = time.perf_counter()
        try:
            res = collector.collect(page, self.timeout_s)
        except Exception as e:  # noqa: BLE001  необроблений виняток – теж результат
            from ..domain import CollectResult
            res = CollectResult(error_type="exception", error_msg=repr(e)[:300])
        t_ms = (time.perf_counter() - t0) * 1000
        usage = mon.stop()
        self._mark_hit(page.url)

        if res.error_type is None and t_ms > self.timeout_s * 1000:
            res.error_type = "timeout"
        if res.error_type is None and not res.records:
            res.error_type = "empty"

        row.update(
            n_found=len(res.records), status_code=res.status_code, error_type=res.error_type or "",
            error_msg=res.error_msg, antibot_hits=res.antibot_hits,
            t_ms=round(t_ms, 2), traffic_mb=round(res.bytes_wire / 1e6, 4), body_mb=round(res.bytes_body / 1e6, 4),
            ram_peak_mb=usage.ram_peak_mb, ram_baseline_mb=usage.ram_baseline_mb, cpu_s=usage.cpu_time_s,
            n_requests=res.n_requests, n_xhr=res.n_xhr, html_size=res.html_size,
            fallback=res.extra.get("fallback", ""), m1_found=res.extra.get("m1_found", ""),
            browser_launch_ms=res.extra.get("browser_launch_ms", ""), monitor_samples=usage.n_samples,
        )

        truth = load_ground_truth(self.gt_dir, page.page_id) if self.gt_dir else None
        technical_ok = res.error_type in (None, "", "empty") and (res.status_code or 0) < 400
        if truth is not None:
            q = evaluate(res.records, truth)
            row.update(q.as_dict())
            row["throughput"] = round(q.tp / (t_ms / 1000), 3) if t_ms > 0 else ""
            row["success"] = bool(technical_ok and q.recall >= SUCCESS_RECALL)
        else:
            row["success"] = ""          # без еталону успішність (умова 3) не визначається

        raw = {"task": {"run_id": task.run_id, "page_id": page.page_id, "method": task.method, "rep": task.rep},
               "measurements": {k: row.get(k) for k in LOG_FIELDS}, "result": res.to_dict()}
        (self.raw_dir / f"{task.run_id}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        return row
