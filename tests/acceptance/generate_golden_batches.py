#!/usr/bin/env python3
"""Generate golden dashboard batch outputs for parity testing.

Usage (from it_latest venv):
    SAE_DASHBOARD_BASELINE_WORKTREE=/path/to/SAEDashboard-7886eaa \
    python tests/acceptance/generate_golden_batches.py

Generates golden batches for both dense_packed and example_aligned families
using the detached baseline worktree's NeuronpediaRunner.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
GOLDEN_BATCHES_DIR = THIS_DIR / "golden_batches"


def _find_baseline_worktree() -> Path:
    env_path = os.environ.get("SAE_DASHBOARD_BASELINE_WORKTREE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    default = Path("/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/baseline_worktrees_20260518/SAEDashboard-7886eaa")
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
    """Run the detached baseline NeuronpediaRunner and return the batch output directory."""
    runner_script = """
import json, os, sys
baseline_root = os.environ["SAE_DASHBOARD_BASELINE_WORKTREE"]
sys.path.insert(0, baseline_root)
from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner, NeuronpediaRunnerConfig

cfg = NeuronpediaRunnerConfig(
    sae_set=os.environ["SAE_SET"],
    sae_path=os.environ["SAE_PATH"],
    np_set_name=os.environ["NP_SET_NAME"],
    from_local_sae=False,
    outputs_dir=os.environ["OUTPUT_DIR"],
    sparsity_threshold=1,
    n_prompts_total=int(os.environ["N_PROMPTS_TOTAL"]),
    n_features_at_a_time=int(os.environ["N_FEATURES"]),
    n_prompts_in_forward_pass=int(os.environ["N_PROMPTS_FWD"]),
    start_batch=0,
    end_batch=int(os.environ["N_BATCHES"]) - 1,
    use_wandb=False,
    shuffle_tokens=False,
)
runner = NeuronpediaRunner(cfg)
runner.run()
batch_dir = os.path.join(cfg.outputs_dir, os.listdir(cfg.outputs_dir)[0])
for i in range(int(os.environ["N_BATCHES"])):
    assert os.path.exists(os.path.join(batch_dir, f"batch-{{i}}.json"))
print(batch_dir)
"""

    script_path = output_dir / "_baseline_runner.py"
    script_path.write_text(runner_script)

    env = os.environ.copy()
    env.update({
        "SAE_DASHBOARD_BASELINE_WORKTREE": str(baseline_worktree),
        "SAE_SET": "gpt2-small-res-jb",
        "SAE_PATH": "blocks.0.hook_resid_pre",
        "NP_SET_NAME": "res-jb",
        "OUTPUT_DIR": str(output_dir / "runner_output"),
        "N_PROMPTS_TOTAL": "64",
        "N_FEATURES": str(n_features),
        "N_PROMPTS_FWD": "16",
        "N_BATCHES": str(n_batches),
    })

    print(f"Running baseline runner ({n_features} features x {n_batches} batches)...")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(baseline_worktree),
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Baseline runner failed with code {result.returncode}")
    batch_dir = Path(result.stdout.strip().splitlines()[-1])
    print(f"  Completed: {batch_dir}")
    return batch_dir


def generate_golden_batches() -> None:
    baseline_worktree = _find_baseline_worktree()
    print(f"Baseline worktree: {baseline_worktree}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        runner_root = tmp / "runner"
        runner_root.mkdir()

        batch_dir = _run_baseline_runner(
            baseline_worktree,
            runner_root,
            n_features=2,
            n_batches=2,
        )

        # Copy to both golden batch families (dense_packed uses Monology/default dataset)
        for family in ["dense_packed", "example_aligned"]:
            family_dir = GOLDEN_BATCHES_DIR / family
            family_dir.mkdir(parents=True, exist_ok=True)

            for i in range(2):
                src = batch_dir / f"batch-{i}.json"
                if src.exists():
                    shutil.copy2(src, family_dir / f"batch-{i}.json")
                    print(f"  [{family}] batch-{i}.json ({src.stat().st_size} bytes)")

            # Copy run_settings if present
            run_settings = runner_root / "runner_output" / "run_settings.json"
            if run_settings.exists():
                shutil.copy2(run_settings, family_dir / "run_settings.json")
                print(f"  [{family}] run_settings.json")

            # Write a basic sae_lens.json metadata
            sae_lens = family_dir / "sae_lens.json"
            sae_lens.write_text(json.dumps({
                "windowing_mode": "concatenate" if family == "dense_packed" else "max-prompt-pad",
                "prompt_windowing_family": "packed_legacy" if family == "dense_packed" else "example_aligned_pad_enabled",
                "effective_context_size": 128 if family == "dense_packed" else 64,
                "context_size": 128 if family == "dense_packed" else 64,
                "tokenizer_name": "gpt2",
                "dataset_path": "monology/pile-uncopyrighted" if family == "dense_packed" else "custom-example-aligned",
            }))
            print(f"  [{family}] sae_lens.json")

    print(f"\nGolden batches generated successfully.")


def main() -> None:
    generate_golden_batches()


if __name__ == "__main__":
    main()
