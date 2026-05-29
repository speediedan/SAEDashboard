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

DETACHED_SLOW_ROOT_ENV = "SAE_DASHBOARD_DETACHED_SLOW_REPLAY_ROOT"
CURRENT_SLOW_ROOT_ENV = "SAE_DASHBOARD_CURRENT_SLOW_REPLAY_ROOT"
CURRENT_FAST_ROOT_ENV = "SAE_DASHBOARD_CURRENT_FAST_REPLAY_ROOT"
BASELINE_WORKTREE_ENV = "SAE_DASHBOARD_BASELINE_WORKTREE_ROOT"

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


@dataclass(frozen=True)
class ReplayRunSpec:
    label: str
    run_root: Path
    run_settings_path: Path
    activation_cache_dir: Path


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

    for minibatch_index in range(REPLAY_MINIBATCHES):
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
def test_rolling_coefficient_update_replay_restores_detached_legacy_cpu_behavior() -> None:
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