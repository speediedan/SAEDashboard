import re
from contextlib import nullcontext

import sae_dashboard.perf_logging as perf_logging


def test_log_perf_event_prefixes_local_timestamp(capsys) -> None:
    perf_logging.log_perf_event("batch_total", batch=3, wall_s=1.25)

    captured = capsys.readouterr().out.strip()

    assert re.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[runner_perf\] event=batch_total batch=3 wall_s=1\.250000$",
        captured,
    )


def test_timed_stage_skips_nested_cuda_synchronization(
    monkeypatch,
) -> None:
    perf_events: list[dict[str, object]] = []
    sync_calls: list[object | None] = []
    event_instances: list[object] = []

    class _FakeEvent:
        def record(self, stream=None) -> None:
            del stream

        def elapsed_time(self, other) -> float:
            del other
            return 1.25

    monkeypatch.setattr(
        perf_logging,
        "log_perf_event",
        lambda event, /, **fields: perf_events.append({"event": event, **fields}),
    )
    monkeypatch.setattr(perf_logging.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        perf_logging.torch.cuda,
        "synchronize",
        lambda device=None: sync_calls.append(device),
    )
    monkeypatch.setattr(perf_logging.torch.cuda, "current_stream", lambda device: device)
    monkeypatch.setattr(
        perf_logging.torch.cuda,
        "Event",
        lambda enable_timing=True: event_instances.append(_FakeEvent()) or event_instances[-1],
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda.nvtx,
        "range_push",
        lambda stage: None,
    )
    monkeypatch.setattr(
        perf_logging.torch.cuda.nvtx,
        "range_pop",
        lambda: None,
    )
    monkeypatch.setattr(
        perf_logging.torch.profiler,
        "record_function",
        lambda stage: nullcontext(),
    )

    with perf_logging.timed_stage(True, "outer", device="cuda"):
        with perf_logging.timed_stage(True, "inner", device="cuda"):
            pass

    assert len(sync_calls) == 2
    assert len(event_instances) == 2

    outer_event = next(
        event
        for event in perf_events
        if event["event"] == "stage_timing" and event["stage"] == "outer"
    )
    inner_event = next(
        event
        for event in perf_events
        if event["event"] == "stage_timing" and event["stage"] == "inner"
    )

    assert outer_event["cuda_ms"] == 1.25
    assert "cuda_ms" not in inner_event