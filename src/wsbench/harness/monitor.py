"""ResourceMonitor – вимірювання пікової пам'яті (M9) і процесорного часу (M10).

Окремий потік кожні 100 мс опитує процес збирача РАЗОМ з усіма дочірніми процесами
(для М3 це процеси Chromium і драйвера Playwright – без них пам'ять була б занижена в рази).

CPU-час: для кожного PID запам'ятовується останнє побачене значення user+system;
підсумок = сума по всіх PID мінус значення на старті. Так враховуються і дочірні процеси,
які встигли завершитися до кінця прогону (браузер закривається всередині прогону).
Обмеження: CPU, витрачений процесом за останні <100 мс перед його завершенням, може
бути не врахований; пік пам'яті між опитуваннями може бути пропущений. Частота
опитування однакова для всіх методів, тому зміщення однакове і на порівняння не впливає.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil


@dataclass
class ResourceUsage:
    ram_peak_mb: float
    ram_baseline_mb: float
    cpu_time_s: float
    n_samples: int
    max_procs: int


class ResourceMonitor:
    def __init__(self, interval_s: float = 0.1, enabled: bool = True):
        self.interval_s = interval_s
        self.enabled = enabled
        self.root = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_start: dict[int, float] = {}
        self._cpu_last: dict[int, float] = {}
        self._peak = 0
        self._baseline = 0
        self._samples = 0
        self._max_procs = 0

    def _tree(self) -> list[psutil.Process]:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except psutil.Error:
            return [self.root]

    def _sample(self) -> int:
        total = 0
        procs = self._tree()
        for p in procs:
            try:
                with p.oneshot():
                    total += p.memory_info().rss
                    ct = p.cpu_times()
                    self._cpu_last[p.pid] = ct.user + ct.system
            except psutil.Error:
                continue
        self._samples += 1
        self._max_procs = max(self._max_procs, len(procs))
        return total

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._peak = max(self._peak, self._sample())

    def start(self) -> None:
        self._cpu_last.clear()
        self._baseline = self._sample()
        self._cpu_start = dict(self._cpu_last)
        self._peak = self._baseline
        self._samples = 0
        self._stop.clear()
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> ResourceUsage:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._peak = max(self._peak, self._sample())
        cpu = sum(v - self._cpu_start.get(pid, 0.0) for pid, v in self._cpu_last.items())
        mb = 1024 * 1024
        return ResourceUsage(
            ram_peak_mb=round(self._peak / mb, 2),
            ram_baseline_mb=round(self._baseline / mb, 2),
            cpu_time_s=round(max(cpu, 0.0), 4),
            n_samples=self._samples,
            max_procs=self._max_procs,
        )
