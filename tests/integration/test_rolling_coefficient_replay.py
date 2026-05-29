import gc
import importlib.util
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einops
import pytest
import torch
from sae_lens import SAE, SkipTranscoder

from sae_dashboard.feature_data_generator import FeatureMaskingContext
from sae_dashboard.neuronpedia.legacy.utils_fns import (
    RollingCorrCoef as LegacyRollingCorrCoef,
)
from sae_dashboard.utils_fns import RollingCorrCoef

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-only.
    resource = None  # type: ignore[assignment]

DETACHED_SLOW_ROOT_ENV = "SAE_DASHBOARD_DETACHED_SLOW_REPLAY_ROOT"
CURRENT_SLOW_ROOT_ENV = "SAE_DASHBOARD_CURRENT_SLOW_REPLAY_ROOT"
CURRENT_FAST_ROOT_ENV = "SAE_DASHBOARD_CURRENT_FAST_REPLAY_ROOT"
BASELINE_WORKTREE_ENV = "SAE_DASHBOARD_BASELINE_WORKTREE_ROOT"
VARIANCE_REPLAY_ENV = "SAE_DASHBOARD_RUN_VARIANCE_REPLAY"
VARIANCE_REPLAY_OUTPUT_ENV = "SAE_DASHBOARD_VARIANCE_REPLAY_OUTPUT"

DEFAULT_DETACHED_SLOW_ROOT = Path(
    "/tmp/np_dashboard_generation_profiles/phase4_activation_encode_rerun_postcommit_20260527/"
    "20260527_235611/phase3-legacy-monology-pretokenized-reduced"
)
DEFAULT_CURRENT_SLOW_ROOT = Path(
    "/tmp/np_dashboard_generation_profiles/phase4_activation_encode_rerun_postcommit_20260527/"
    "20260528_000103/phase4-current-legacy-monology-pretokenized-reduced"
)
DEFAULT_CURRENT_FAST_ROOT = Path(
    "/tmp/np_dashboard_generation_profiles/phase4_activation_encode_rerun_followup_postcommit_20260528/"
    "20260528_005036/phase4-current-legacy-monology-pretokenized-reduced"
)
DEFAULT_BASELINE_WORKTREE_ROOT = Path(
    "/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/"
    "baseline_worktrees_20260518/SAEDashboard-7886eaa"
)
REPLAY_MINIBATCHES = 3
REPLAY_REPEATS = 6
VARIANCE_REPLAY_REPEATS = 12
VARIANCE_REPLAY_OUTPUT_DEFAULT = Path(
    "/tmp/rolling_coefficient_update_variance_replay.json"
)
DEFAULT_VARIANCE_REPLAY_CASES = (
    (
        "rte_fast_current",
        Path(
            "/tmp/np_dashboard_generation_profiles/phase4_threeway_perf_logged_9a83ccb_20260529/"
            "current_legacy_rte/20260529_192817/phase4-current-legacy-rte-pretokenized-reduced"
        ),
        (1,),
    ),
    (
        "rte_slow_current",
        Path(
            "/tmp/np_dashboard_generation_profiles/phase4_threeway_perf_logged_c2250fa_20260529/"
            "current_legacy_rte/20260529_181928/phase4-current-legacy-rte-pretokenized-reduced"
        ),
        (1,),
    ),
    (
        "monology_new_current",
        Path(
            "/tmp/np_dashboard_generation_profiles/phase4_threeway_perf_logged_9a83ccb_20260529/"
            "current_legacy_monology/20260529_193858/phase4-current-legacy-monology-pretokenized-reduced"
        ),
        (1,),
    ),
    (
        "monology_old_current",
        Path(
            "/tmp/np_dashboard_generation_profiles/phase4_threeway_perf_logged_c2250fa_20260529/"
            "current_legacy_monology/20260529_183019/phase4-current-legacy-monology-pretokenized-reduced"
        ),
        (1,),
    ),
)


@dataclass(frozen=True)
class ReplayRunSpec:
    label: str
    run_root: Path
    run_settings_path: Path
    activation_cache_dir: Path
    minibatch_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReplayInputs:
    model_acts: torch.Tensor
    feature_acts: torch.Tensor


@dataclass(frozen=True)
class TimingSummary:
    label: str
    current_median_ms: float
    compat_median_ms: float
    baseline_median_ms: float
    current_iteration_medians_ms: tuple[float, ...]
    compat_iteration_medians_ms: tuple[float, ...]
    baseline_iteration_medians_ms: tuple[float, ...]


def _read_optional_path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, str(default)))


def _find_single_nested_path(run_root: Path, target_name: str) -> Path | None:
    matches = sorted(run_root.rglob(target_name))
    if not matches:
        return None
    return matches[0]


def _resolve_replay_specs() -> list[ReplayRunSpec]:
    candidate_roots = [
        (
            "detached_slow",
            _read_optional_path(
                DETACHED_SLOW_ROOT_ENV,
                DEFAULT_DETACHED_SLOW_ROOT,
            ),
        ),
        (
            "current_slow",
            _read_optional_path(
                CURRENT_SLOW_ROOT_ENV,
                DEFAULT_CURRENT_SLOW_ROOT,
            ),
        ),
        (
            "current_fast",
            _read_optional_path(
                CURRENT_FAST_ROOT_ENV,
                DEFAULT_CURRENT_FAST_ROOT,
            ),
        ),
    ]
    missing = [str(run_root) for _, run_root in candidate_roots if not run_root.exists()]
    if missing:
        pytest.skip(
            "Replay roots are not available: " + ", ".join(missing)
        )

    specs: list[ReplayRunSpec] = []
    for label, run_root in candidate_roots:
        run_settings_path = _find_single_nested_path(run_root, "run_settings.json")
        activation_cache_dir = _find_single_nested_path(run_root, "_activation_cache")
        if run_settings_path is None or activation_cache_dir is None:
            continue
        specs.append(
            ReplayRunSpec(
                label=label,
                run_root=run_root,
                run_settings_path=run_settings_path,
                activation_cache_dir=activation_cache_dir,
            )
        )

    if len(specs) < 2:
        pytest.skip("Need at least two replay roots with cached minibatches")
    return specs


def _resolve_variance_replay_specs() -> list[ReplayRunSpec]:
    specs: list[ReplayRunSpec] = []
    missing: list[str] = []
    for label, run_root, minibatch_indices in DEFAULT_VARIANCE_REPLAY_CASES:
        if not run_root.exists():
            missing.append(str(run_root))
            continue
        run_settings_path = _find_single_nested_path(run_root, "run_settings.json")
        activation_cache_dir = _find_single_nested_path(run_root, "_activation_cache")
        if run_settings_path is None or activation_cache_dir is None:
            missing.append(str(run_root))
            continue
        specs.append(
            ReplayRunSpec(
                label=label,
                run_root=run_root,
                run_settings_path=run_settings_path,
                activation_cache_dir=activation_cache_dir,
                minibatch_indices=minibatch_indices,
            )
        )
    if missing:
        pytest.skip("Variance replay roots are not available: " + ", ".join(missing))
    return specs


def _load_run_settings(run_root: Path) -> dict[str, Any]:
    return json.loads(run_root.read_text(encoding="utf-8"))


def _load_encoder(run_settings: dict[str, Any]) -> torch.nn.Module:
    sae_dtype = getattr(torch, run_settings["sae_dtype"])
    sae_device = run_settings["sae_device"]

    if run_settings.get("use_skip_transcoder"):
        encoder = SkipTranscoder.from_pretrained(
            release=run_settings["sae_set"],
            sae_id=run_settings["sae_path"],
            device=sae_device,
        )
    else:
        encoder = SAE.from_pretrained(
            release=run_settings["sae_set"],
            sae_id=run_settings["sae_path"],
            device=sae_device,
        )

    return encoder.to(dtype=sae_dtype)


def _load_baseline_utils_module() -> Any:
    baseline_root = _read_optional_path(
        BASELINE_WORKTREE_ENV,
        DEFAULT_BASELINE_WORKTREE_ROOT,
    )
    utils_path = baseline_root / "sae_dashboard" / "utils_fns.py"
    if not utils_path.exists():
        pytest.skip(f"Baseline utils module not found at {utils_path}")
    spec = importlib.util.spec_from_file_location(
        "baseline_sae_dashboard_utils_fns",
        utils_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline utils module from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_replay_inputs(spec: ReplayRunSpec) -> list[ReplayInputs]:
    run_settings = _load_run_settings(spec.run_settings_path)
    encoder = _load_encoder(run_settings)
    feature_indices = list(range(run_settings["n_features_at_a_time"]))
    hook_name = f"blocks.{run_settings['layer']}.hook_mlp_in"
    replay_inputs: list[ReplayInputs] = []
    minibatch_indices = spec.minibatch_indices or tuple(range(REPLAY_MINIBATCHES))

    for minibatch_index in minibatch_indices:
        cache_path = spec.activation_cache_dir / f"model_activations_{minibatch_index}.pt"
        activation_dict = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        model_acts = activation_dict[hook_name].to(run_settings["sae_device"])

        with torch.inference_mode():
            with FeatureMaskingContext(encoder, feature_indices):
                feature_acts = encoder.encode(model_acts).to(
                    getattr(torch, run_settings["sae_dtype"])
                )

        replay_inputs.append(
            ReplayInputs(model_acts=model_acts, feature_acts=feature_acts)
        )

    return replay_inputs


def _measure_update_ms(
    replay_input: ReplayInputs,
    *,
    repeats: int,
    corrcoef_cls: type[Any],
) -> tuple[float, ...]:
    timings_ms: list[float] = []
    feature_indices = list(range(replay_input.feature_acts.shape[-1]))

    for _ in range(repeats):
        corrcoef_neurons = corrcoef_cls(
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        corrcoef_encoder = corrcoef_cls(
            indices=feature_indices,
            with_self=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        start_time = time.perf_counter()
        feature_acts_by_feature = einops.rearrange(
            replay_input.feature_acts,
            "batch seq feats -> feats (batch seq)",
        )
        corrcoef_neurons.update(
            feature_acts_by_feature,
            einops.rearrange(
                replay_input.model_acts,
                "batch seq d_in -> d_in (batch seq)",
            ),
        )
        corrcoef_encoder.update(
            feature_acts_by_feature,
            feature_acts_by_feature,
        )
        timings_ms.append((time.perf_counter() - start_time) * 1000)

    return tuple(timings_ms)


def _measure_baseline_update_ms(
    replay_input: ReplayInputs,
    *,
    repeats: int,
    baseline_utils_module: Any,
) -> tuple[float, ...]:
    timings_ms: list[float] = []
    feature_indices = list(range(replay_input.feature_acts.shape[-1]))
    baseline_corrcoef_cls = baseline_utils_module.RollingCorrCoef

    for _ in range(repeats):
        corrcoef_neurons = baseline_corrcoef_cls(
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        corrcoef_encoder = baseline_corrcoef_cls(
            indices=feature_indices,
            with_self=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        start_time = time.perf_counter()
        corrcoef_neurons.update(
            einops.rearrange(
                replay_input.feature_acts,
                "batch seq feats -> feats (batch seq)",
            ),
            einops.rearrange(
                replay_input.model_acts,
                "batch seq d_in -> d_in (batch seq)",
            ),
        )
        corrcoef_encoder.update(
            einops.rearrange(
                replay_input.feature_acts,
                "batch seq feats -> feats (batch seq)",
            ),
            einops.rearrange(
                replay_input.feature_acts,
                "batch seq feats -> feats (batch seq)",
            ),
        )
        timings_ms.append((time.perf_counter() - start_time) * 1000)

    return tuple(timings_ms)


def _rusage_snapshot() -> dict[str, float | int]:
    if resource is None:
        return {
            "utime_s": 0.0,
            "stime_s": 0.0,
            "maxrss_kb": 0,
            "minor_faults": 0,
            "major_faults": 0,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": 0,
        }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "utime_s": usage.ru_utime,
        "stime_s": usage.ru_stime,
        "maxrss_kb": usage.ru_maxrss,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _rusage_delta(
    start: dict[str, float | int],
    end: dict[str, float | int],
) -> dict[str, float | int]:
    return {key: end[key] - start[key] for key in start}


def _summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _measure_update_trace(
    replay_input: ReplayInputs,
    *,
    repeats: int,
    corrcoef_cls: type[Any],
) -> list[dict[str, Any]]:
    feature_indices = list(range(replay_input.feature_acts.shape[-1]))
    feature_acts_by_feature = einops.rearrange(
        replay_input.feature_acts,
        "batch seq feats -> feats (batch seq)",
    )
    model_acts_by_dim = einops.rearrange(
        replay_input.model_acts,
        "batch seq d_in -> d_in (batch seq)",
    )
    samples: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        gc.collect()
        corrcoef_neurons = corrcoef_cls(
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        corrcoef_encoder = corrcoef_cls(
            indices=feature_indices,
            with_self=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        usage_before = _rusage_snapshot()
        process_time_before = time.process_time()
        start_time = time.perf_counter()
        corrcoef_neurons.update(feature_acts_by_feature, model_acts_by_dim)
        corrcoef_encoder.update(feature_acts_by_feature, feature_acts_by_feature)
        wall_ms = (time.perf_counter() - start_time) * 1000
        process_time_ms = (time.process_time() - process_time_before) * 1000
        usage_after = _rusage_snapshot()
        samples.append(
            {
                "repeat_index": repeat_index,
                "wall_ms": wall_ms,
                "process_time_ms": process_time_ms,
                "rusage_delta": _rusage_delta(usage_before, usage_after),
            }
        )

    return samples


def _variance_replay_case_report(spec: ReplayRunSpec) -> list[dict[str, Any]]:
    run_settings = _load_run_settings(spec.run_settings_path)
    reports: list[dict[str, Any]] = []
    for minibatch_index, replay_input in zip(
        spec.minibatch_indices,
        _load_replay_inputs(spec),
        strict=True,
    ):
        implementation_reports = {}
        for implementation, corrcoef_cls in (
            ("current", RollingCorrCoef),
            ("legacy_compat", LegacyRollingCorrCoef),
        ):
            samples = _measure_update_trace(
                replay_input,
                repeats=VARIANCE_REPLAY_REPEATS,
                corrcoef_cls=corrcoef_cls,
            )
            implementation_reports[implementation] = {
                "summary_ms": _summarize_values(
                    [sample["wall_ms"] for sample in samples]
                ),
                "process_time_summary_ms": _summarize_values(
                    [sample["process_time_ms"] for sample in samples]
                ),
                "samples": samples,
            }
        reports.append(
            {
                "label": spec.label,
                "run_root": str(spec.run_root),
                "minibatch_index": minibatch_index,
                "run_settings": {
                    "n_prompts_in_forward_pass": run_settings.get("n_prompts_in_forward_pass"),
                    "n_tokens_in_prompt": run_settings.get("n_tokens_in_prompt"),
                    "n_features_at_a_time": run_settings.get("n_features_at_a_time"),
                    "sae_device": run_settings.get("sae_device"),
                    "sae_dtype": run_settings.get("sae_dtype"),
                    "correlation_accumulation_device": run_settings.get("correlation_accumulation_device"),
                    "layer": run_settings.get("layer"),
                },
                "model_acts_shape": list(replay_input.model_acts.shape),
                "feature_acts_shape": list(replay_input.feature_acts.shape),
                "feature_acts_stride": list(replay_input.feature_acts.stride()),
                "model_acts_stride": list(replay_input.model_acts.stride()),
                "implementations": implementation_reports,
            }
        )
    return reports


def _runtime_context() -> dict[str, Any]:
    cpu_affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        cpu_affinity = sorted(os.sched_getaffinity(0))
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_affinity": cpu_affinity,
    }


def _summarize_timings(
    spec: ReplayRunSpec,
    replay_inputs: list[ReplayInputs],
    baseline_utils_module: Any,
) -> TimingSummary:
    current_iteration_medians_ms = tuple(
        statistics.median(
            _measure_update_ms(
                replay_input,
                repeats=REPLAY_REPEATS,
                corrcoef_cls=RollingCorrCoef,
            )
        )
        for replay_input in replay_inputs
    )
    compat_iteration_medians_ms = tuple(
        statistics.median(
            _measure_update_ms(
                replay_input,
                repeats=REPLAY_REPEATS,
                corrcoef_cls=LegacyRollingCorrCoef,
            )
        )
        for replay_input in replay_inputs
    )
    baseline_iteration_medians_ms = tuple(
        statistics.median(
            _measure_baseline_update_ms(
                replay_input,
                repeats=REPLAY_REPEATS,
                baseline_utils_module=baseline_utils_module,
            )
        )
        for replay_input in replay_inputs
    )
    return TimingSummary(
        label=spec.label,
        current_median_ms=statistics.median(current_iteration_medians_ms),
        compat_median_ms=statistics.median(compat_iteration_medians_ms),
        baseline_median_ms=statistics.median(baseline_iteration_medians_ms),
        current_iteration_medians_ms=current_iteration_medians_ms,
        compat_iteration_medians_ms=compat_iteration_medians_ms,
        baseline_iteration_medians_ms=baseline_iteration_medians_ms,
    )


def _format_summary_report(summaries: list[TimingSummary]) -> str:
    report_lines = ["rolling_coefficient_update replay summary:"]
    for summary in summaries:
        report_lines.append(
            " - "
            f"{summary.label}: current median={summary.current_median_ms:.1f} ms "
            f"from {list(summary.current_iteration_medians_ms)}; "
            f"compat median={summary.compat_median_ms:.1f} ms "
            f"from {list(summary.compat_iteration_medians_ms)}; "
            f"baseline median={summary.baseline_median_ms:.1f} ms "
            f"from {list(summary.baseline_iteration_medians_ms)}"
        )
    return "\n".join(report_lines)


def _paired_delta_median_ms(
    lhs: tuple[float, ...],
    rhs: tuple[float, ...],
) -> float:
    return statistics.median(
        abs(lhs_value - rhs_value)
        for lhs_value, rhs_value in zip(lhs, rhs, strict=True)
    )


def _compat_matches_baseline_band(summary: TimingSummary) -> bool:
    if abs(summary.compat_median_ms - summary.baseline_median_ms) <= (
        summary.baseline_median_ms * 0.12
    ):
        return True

    baseline_min_ms = min(summary.baseline_iteration_medians_ms)
    baseline_max_ms = max(summary.baseline_iteration_medians_ms)
    return baseline_min_ms * 0.97 <= summary.compat_median_ms <= baseline_max_ms * 1.03


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Replay timing reconstruction requires CUDA for the saved SAE inputs",
)
def test_rolling_coefficient_update_replay_restores_legacy_cpu_behavior() -> None:
    replay_specs = _resolve_replay_specs()
    baseline_utils_module = _load_baseline_utils_module()
    summaries = [
        _summarize_timings(
            spec,
            _load_replay_inputs(spec),
            baseline_utils_module,
        )
        for spec in replay_specs
    ]
    report = _format_summary_report(summaries)
    print(report)

    assert all(_compat_matches_baseline_band(summary) for summary in summaries), report


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Replay timing reconstruction requires CUDA for the saved SAE inputs",
)
def test_rolling_coefficient_update_variance_replay_probe() -> None:
    if os.environ.get(VARIANCE_REPLAY_ENV) != "1":
        pytest.skip(f"Set {VARIANCE_REPLAY_ENV}=1 to run the opt-in variance replay probe")

    reports: list[dict[str, Any]] = []
    for spec in _resolve_variance_replay_specs():
        reports.extend(_variance_replay_case_report(spec))

    output = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_context": _runtime_context(),
        "cases": reports,
    }
    output_path = Path(
        os.environ.get(VARIANCE_REPLAY_OUTPUT_ENV, str(VARIANCE_REPLAY_OUTPUT_DEFAULT))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

    assert all(
        implementation_report["summary_ms"]["median"] > 0
        for case_report in reports
        for implementation_report in case_report["implementations"].values()
    )