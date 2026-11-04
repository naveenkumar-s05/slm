"""Small shared helpers: config loading, seeding, IO and resource metrics."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": props.name,
                "gpu_vram_gb": round(props.total_memory / 1024**3, 2),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def cuda_memory_mb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


class ResourceMonitor:
    """Sample process RSS and CUDA memory in a background thread."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss / 1024**2
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.peak_rss_mb = self._process.memory_info().rss / 1024**2
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        result = {"peak_process_rss_mb": round(self.peak_rss_mb, 1)}
        result.update({f"inference_{key}": value for key, value in cuda_memory_mb().items()})
        return result
