import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import torch

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix platforms
    resource = None

_TIMED_STAGE_DEPTH: ContextVar[int] = ContextVar("timed_stage_depth", default=0)


def _local_log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def _format_perf_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def log_perf_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(
        f"{_local_log_timestamp()} [runner_perf] "
        + " ".join(
            f"{key}={_format_perf_value(value)}" for key, value in payload.items()
        ),
        flush=True,
    )


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


def runtime_snapshot() -> dict[str, Any]:
    snapshot = cpu_snapshot()
    snapshot["torch_num_threads"] = torch.get_num_threads()
    try:
        snapshot["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except RuntimeError:
        pass
    if hasattr(os, "sched_getaffinity"):
        try:
            snapshot["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return snapshot


def rusage_snapshot() -> dict[str, float | int]:
    if resource is None:
        return {}

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def rusage_delta(
    start: dict[str, float | int],
    end: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in sorted(set(start) | set(end))
    }


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
    return {
        key: end.get(key, 0) - start.get(key, 0)
        for key in sorted(set(start) | set(end))
    }


@contextmanager
def temporary_torch_num_threads(num_threads: int | None) -> Iterator[None]:
    if num_threads is None:
        yield
        return
    if num_threads < 1:
        raise ValueError("temporary_torch_num_threads requires num_threads >= 1")

    previous_num_threads = torch.get_num_threads()
    if num_threads == previous_num_threads:
        yield
        return

    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_num_threads)


@contextmanager
def timed_stage(
    enabled: bool,
    stage: str,
    *,
    device: str | torch.device | None = None,
    capture_runtime_metrics: bool = False,
    **fields: Any,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    stage_depth = _TIMED_STAGE_DEPTH.get()
    token = _TIMED_STAGE_DEPTH.set(stage_depth + 1)
    torch_device = torch.device(device) if device is not None else None
    use_cuda_events = (
        stage_depth == 0
        and
        torch_device is not None
        and torch_device.type == "cuda"
        and torch.cuda.is_available()
    )
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda_events else None

    if use_cuda_events and start_event is not None:
        torch.cuda.synchronize(torch_device)
        start_event.record(torch.cuda.current_stream(torch_device))
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(stage)
    start_runtime = runtime_snapshot() if capture_runtime_metrics else None
    start_io = process_io_snapshot() if capture_runtime_metrics else None
    start_rusage = rusage_snapshot() if capture_runtime_metrics else None
    start_process_time = time.process_time() if capture_runtime_metrics else None
    start_time = time.perf_counter()
    try:
        with torch.profiler.record_function(stage):
            yield
    finally:
        try:
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
            if capture_runtime_metrics:
                end_runtime = runtime_snapshot()
                end_io = process_io_snapshot()
                end_rusage = rusage_snapshot()
                if start_process_time is not None:
                    log_fields["process_time_s"] = time.process_time() - start_process_time
                if start_runtime is not None:
                    log_fields["runtime_start"] = start_runtime
                log_fields["runtime_end"] = end_runtime
                if start_rusage is not None:
                    runtime_rusage_delta = rusage_delta(start_rusage, end_rusage)
                    if runtime_rusage_delta:
                        log_fields["rusage_delta"] = runtime_rusage_delta
                if start_io is not None:
                    runtime_io_delta = io_delta(start_io, end_io)
                    if runtime_io_delta:
                        log_fields["process_io_delta"] = runtime_io_delta
            log_perf_event("stage_timing", **log_fields)
        finally:
            _TIMED_STAGE_DEPTH.reset(token)


@contextmanager
def elapsed_timer() -> Iterator[dict[str, float]]:
    timing: dict[str, float] = {}
    start_time = time.perf_counter()
    try:
        yield timing
    finally:
        timing["wall_s"] = time.perf_counter() - start_time
