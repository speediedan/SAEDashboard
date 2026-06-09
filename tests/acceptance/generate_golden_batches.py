#!/usr/bin/env python3
"""Generate golden dashboard batch outputs for parity testing.

Usage (from it_latest venv):
    cd /home/speediedan/repos/SAEDashboard
    SAE_DASHBOARD_BASELINE_WORKTREE=/path/to/SAEDashboard-7886eaa \
    python tests/acceptance/generate_golden_batches.py

Generates golden batches for both dense_packed and example_aligned families
using the detached baseline worktree's NeuronpediaRunner.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
GOLDEN_BATCHES_DIR = THIS_DIR / "golden_batches"


def _find_baseline_worktree() -> Path:
    env_path = os.environ.get("SAE_DASHBOARD_BASELINE_WORKTREE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    default = Path(
        "/mnt/cache_extended/speediedan/.cache/huggingface/"
        "interpretune/neuronpedia/baseline_worktrees_20260518/SAEDashboard-7886eaa"
    )
    if default.exists():
        return default
    raise RuntimeError(
        "Set SAE_DASHBOARD_BASELINE_WORKTREE to the detached baseline worktree path."
    )


def _run_baseline_runner(
    baseline_worktree: Path,
    output_dir: Path,
    *,
    n_features: int = 2,
    n_batches: int = 2,
) -> Path:
    """Run the detached baseline NeuronpediaRunner directly (no subprocess)."""
    # Add the baseline worktree to sys.path so we can import its SAEDashboard
    if str(baseline_worktree) not in sys.path:
        sys.path.insert(0, str(baseline_worktree))

    # The worktree uses the SAELens from the current venv (same venv)
    from sae_dashboard.neuronpedia.neuronpedia_runner import (
        NeuronpediaRunner,
        NeuronpediaRunnerConfig,
    )

    cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.0.hook_resid_pre",
        np_set_name="res-jb",
        from_local_sae=False,
        outputs_dir=str(output_dir / "runner_output"),
        sparsity_threshold=1,
        n_prompts_total=64,
        n_features_at_a_time=n_features,
        n_prompts_in_forward_pass=16,
        start_batch=0,
        end_batch=n_batches - 1,
        use_wandb=False,
        shuffle_tokens=False,
    )

    runner = NeuronpediaRunner(cfg)
    runner.run()

    # The runner puts batch-*.json files directly in the output subdirectory.
    # cfg.outputs_dir / first_subdir / batch-*.json
    outputs_root = Path(cfg.outputs_dir)
    subdirs = [e for e in os.listdir(str(outputs_root))
               if os.path.isdir(str(outputs_root / e))]
    if subdirs:
        batch_dir = outputs_root / subdirs[0]
    else:
        # Batch files may be directly in outputs_root
        batch_dir = outputs_root

    # Verify batch files exist
    for i in range(n_batches):
        bp = batch_dir / f"batch-{i}.json"
        assert bp.exists(), f"Missing {bp}"

    return batch_dir


def generate_golden_batches() -> None:
    baseline_worktree = _find_baseline_worktree()
    print(f"Baseline worktree: {baseline_worktree}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        batch_dir = _run_baseline_runner(
            baseline_worktree,
            tmp,
            n_features=2,
            n_batches=2,
        )

        # Copy to both golden batch families
        for family in ["dense_packed", "example_aligned"]:
            family_dir = GOLDEN_BATCHES_DIR / family
            family_dir.mkdir(parents=True, exist_ok=True)

            for i in range(2):
                src = batch_dir / f"batch-{i}.json"
                if src.exists():
                    shutil.copy2(src, family_dir / f"batch-{i}.json")
                    print(f"  [{family}] batch-{i}.json ({src.stat().st_size} bytes)")

            # Copy run_settings if present
            run_settings_src = batch_dir / "run_settings.json"
            if run_settings_src.exists():
                shutil.copy2(run_settings_src, family_dir / "run_settings.json")
                print(f"  [{family}] run_settings.json")

            # Write minimal sae_lens.json metadata
            sae_lens = family_dir / "sae_lens.json"
            sae_lens.write_text(json.dumps({
                "windowing_mode": (
                    "concatenate" if family == "dense_packed" else "max-prompt-pad"
                ),
                "prompt_windowing_family": (
                    "packed_legacy" if family == "dense_packed"
                    else "example_aligned_pad_enabled"
                ),
                "effective_context_size": (
                    128 if family == "dense_packed" else 64
                ),
                "context_size": 128 if family == "dense_packed" else 64,
                "tokenizer_name": "gpt2",
                "dataset_path": (
                    "monology/pile-uncopyrighted" if family == "dense_packed"
                    else "custom-example-aligned"
                ),
            }))
            print(f"  [{family}] sae_lens.json")

    print("\nGolden batches generated successfully.")


if __name__ == "__main__":
    generate_golden_batches()
