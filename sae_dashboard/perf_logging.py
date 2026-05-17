import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _format_perf_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def log_perf_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print("[runner_perf] " + " ".join(f"{key}={_format_perf_value(value)}" for key, value in payload.items()), flush=True)


def cpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        load1, load5, load15 = os.getloadavg()
        snapshot.update({"load1": load1, "load5": load5, "load15": load15})
    except OSError:
        pass

    loadavg_path = Path("/proc/loadavg")
    if loadavg_path.exists():
        parts = loadavg_path.read_text(encoding="utf-8", errors="replace").split()
        if len(parts) >= 4 and "/" in parts[3]:
            runnable_threads, total_threads = parts[3].split("/", 1)
            snapshot["runnable_threads"] = int(runnable_threads)
            snapshot["system_threads"] = int(total_threads)

    task_dir = Path("/proc/self/task")
    if task_dir.exists():
        snapshot["process_threads"] = sum(1 for _ in task_dir.iterdir())
    return snapshot


def process_io_snapshot() -> dict[str, int]:
    io_path = Path("/proc/self/io")
    if not io_path.exists():
        return {}

    snapshot: dict[str, int] = {}
    for line in io_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        if raw_value.isdigit():
            snapshot[key] = int(raw_value)
    return snapshot


def io_delta(start: dict[str, int], end: dict[str, int]) -> dict[str, int]:
    return {key: end.get(key, 0) - start.get(key, 0) for key in sorted(set(start) | set(end))}


@contextmanager
def timed_stage(
    enabled: bool,
    stage: str,
    *,
    device: str | torch.device | None = None,
    **fields: Any,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    torch_device = torch.device(device) if device is not None else None
    use_cuda_events = torch_device is not None and torch_device.type == "cuda" and torch.cuda.is_available()
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None

    if use_cuda_events and start_event is not None:
        torch.cuda.synchronize(torch_device)
        start_event.record(torch.cuda.current_stream(torch_device))
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(stage)
    start_time = time.perf_counter()
    try:
        with torch.profiler.record_function(stage):
            yield
    finally:
        wall_seconds = time.perf_counter() - start_time
        cuda_ms: float | None = None
        if use_cuda_events and start_event is not None and end_event is not None:
            end_event.record(torch.cuda.current_stream(torch_device))
            torch.cuda.synchronize(torch_device)
            cuda_ms = float(start_event.elapsed_time(end_event))
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        log_fields = dict(fields)
        log_fields.update({"stage": stage, "wall_s": wall_seconds})
        if cuda_ms is not None:
            log_fields["cuda_ms"] = cuda_ms
        log_perf_event("stage_timing", **log_fields)


@contextmanager
def elapsed_timer() -> Iterator[dict[str, float]]:
    timing: dict[str, float] = {}
    start_time = time.perf_counter()
    try:
        yield timing
    finally:
        timing["wall_s"] = time.perf_counter() - start_time