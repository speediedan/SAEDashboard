import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_dashboard.neuronpedia.neuronpedia_runner import NeuronpediaRunner
from sae_dashboard.neuronpedia.neuronpedia_runner_config import NeuronpediaRunnerConfig


@pytest.fixture
def runner_config() -> NeuronpediaRunnerConfig:
    return NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=256,
        n_tokens_in_prompt=128,
        huggingface_dataset_path="monology/pile-uncopyrighted",
    )


@pytest.fixture
def neuronpedia_runner(
    runner_config: NeuronpediaRunnerConfig, model: HookedTransformer
) -> NeuronpediaRunner:
    runner = NeuronpediaRunner(runner_config)
    runner.model = model  # type: ignore
    return runner


def test_generate_tokens_no_duplicates(neuronpedia_runner: NeuronpediaRunner) -> None:
    tokens = neuronpedia_runner.generate_tokens(
        neuronpedia_runner.activations_store, n_prompts=256
    )
    assert tokens.shape == (256, neuronpedia_runner.cfg.n_tokens_in_prompt)
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert len(torch.unique(tokens_cpu, dim=0)) == 256


def test_get_tokens_no_duplicates(
    neuronpedia_runner: NeuronpediaRunner, tmp_path: Path
) -> None:
    neuronpedia_runner.cfg.outputs_dir = str(tmp_path)
    tokens = neuronpedia_runner.get_tokens()
    assert tokens.shape == (
        neuronpedia_runner.cfg.n_prompts_total,
        neuronpedia_runner.cfg.n_tokens_in_prompt,
    )
    # Move to CPU before checking uniqueness
    tokens_cpu = tokens.cpu()
    assert (
        len(torch.unique(tokens_cpu, dim=0)) == neuronpedia_runner.cfg.n_prompts_total
    )


def test_materialize_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = Dataset.from_dict(
        {"input_ids": [list(range(128)), list(range(128, 256)), list(range(256, 384))]}
    )
    pretokenized_path = tmp_path / "rte_tokens"

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=2,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
    )

    materialized_dataset = runner._materialize_prompt_dataset()

    assert isinstance(materialized_dataset, Dataset)
    assert len(materialized_dataset) == 2
    assert materialized_dataset.column_names == ["input_ids"]


def test_materialize_structured_dataset_uses_supplied_text_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_dict(
        {"prompt": ["Already rendered prompt."], "other": ["ignored"]}
    )

    def fake_load_dataset(
        path: str,
        config_name: str | None,
        *,
        split: str,
        streaming: bool,
    ) -> Dataset:
        assert path == "aps/super_glue"
        assert config_name == "rte"
        assert split == "train"
        assert streaming is True
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_dataset",
        fake_load_dataset,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=1,
        huggingface_dataset_path="aps/super_glue",
        huggingface_dataset_config_name="rte",
        huggingface_dataset_split="train",
        huggingface_dataset_text_field="prompt",
    )

    materialized_dataset = runner._materialize_prompt_dataset()

    assert isinstance(materialized_dataset, Dataset)
    assert materialized_dataset[0]["text"] == "Already rendered prompt."


def test_create_output_directory_sanitizes_model_id(tmp_path: Path) -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_10"),
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.10.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))
    runner.np_sae_id_suffix = None

    output_dir = Path(runner.create_output_directory())

    assert output_dir.parent == tmp_path / "layer_10"
    assert (
        output_dir.name
        == "google_gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.10.hook_mlp_in_262144"
    )


def test_resolved_neuronpedia_set_name_appends_suffix() -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir="test_outputs",
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )
    runner.np_sae_id_suffix = "phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"

    resolved_name = runner._resolved_neuronpedia_set_name()

    assert (
        resolved_name
        == "gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    )


def test_resolved_columnar_activation_copy_ids_include_suffix() -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir="test_outputs",
        np_set_name="gemmascope-2-transcoder-262k-rte",
    )
    runner.np_sae_id_suffix = "phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    runner.layer = 9

    assert runner._resolved_columnar_activation_copy_layer() == (
        "9-gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514"
    )
    assert runner._resolved_columnar_activation_copy_id_prefix() == (
        "9-gemmascope-2-transcoder-262k-rte__phase1-pr-clean-lazy-parquet-importwarmfix-l9-20260514-activation"
    )


def test_setup_output_directory_stages_shared_tokens_file(tmp_path: Path) -> None:
    shared_tokens_file = tmp_path / "shared_tokens.pt"
    expected_tokens = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)
    torch.save(expected_tokens, shared_tokens_file)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_10"),
        n_prompts_total=2,
        shared_tokens_file=str(shared_tokens_file),
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.10.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()

    assert torch.equal(tokens, expected_tokens)
    assert Path(runner.cfg.outputs_dir, "tokens_2.pt").exists()


def test_load_prompt_bucket_schedule_uses_selected_bucket_configs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "layer_10"
    output_dir.mkdir(parents=True)
    tokens = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        dtype=torch.long,
    )
    torch.save(tokens, output_dir / "tokens_6.pt")
    torch.save(
        torch.tensor([32, 64, 65, 100, 128, 200], dtype=torch.int32),
        output_dir / "tokens_6.effective_lengths.pt",
    )
    schedule_path = tmp_path / "selected_bucket_configs.json"
    schedule_path.write_text(
        json.dumps(
            {
                "buckets": [
                    {
                        "bucket_ceiling": 64,
                        "prompt_count": 2,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 2,
                            "primary_acts_batch_size": 2,
                        },
                    },
                    {
                        "bucket_ceiling": 128,
                        "prompt_count": 3,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 2,
                            "primary_acts_batch_size": 1,
                        },
                    },
                    {
                        "bucket_ceiling": 319,
                        "prompt_count": 1,
                        "selected_config": {
                            "n_prompts_in_forward_pass": 4,
                            "primary_acts_batch_size": 4,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=6,
        shared_tokens_file=str(tmp_path / "shared_tokens.pt"),
        prompt_bucket_schedule_file=str(schedule_path),
    )

    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 64, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [4], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [5], "seq_length": 319, "primary_acts_batch_size": 4},
    ]


def test_load_prompt_bucket_schedule_can_auto_bucket_from_effective_lengths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "layer_10"
    output_dir.mkdir(parents=True)
    tokens = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
            [28, 29, 30],
            [31, 32, 33],
            [34, 35, 36],
        ],
        dtype=torch.long,
    )
    torch.save(tokens, output_dir / "tokens_12.pt")
    torch.save(
        torch.tensor(
            [40, 50, 60, 64, 64, 64, 65, 70, 80, 90, 110, 120], dtype=torch.int32
        ),
        output_dir / "tokens_12.effective_lengths.pt",
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_10_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=12,
        n_tokens_in_prompt=128,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        shared_tokens_file=str(tmp_path / "shared_tokens.pt"),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(64,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )

    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert schedule == [
        {
            "prompt_indices": [0, 1, 2, 3],
            "seq_length": 64,
            "primary_acts_batch_size": 2,
        },
        {"prompt_indices": [4, 5], "seq_length": 64, "primary_acts_batch_size": 2},
        {"prompt_indices": [6, 7], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [8, 9], "seq_length": 128, "primary_acts_batch_size": 1},
        {"prompt_indices": [10, 11], "seq_length": 128, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_can_generate_shared_sidecars_from_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [21, 22, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ],
            "attention_mask": [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    output_dir = tmp_path / "layer_9"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    shared_tokens_file = pretokenized_path / "tokens_4.pt"
    assert runner.cfg.shared_tokens_file == str(shared_tokens_file)
    assert shared_tokens_file.is_file()
    assert (pretokenized_path / "tokens_4.effective_lengths.pt").is_file()
    assert (
        output_dir
        / "google_gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.9.hook_mlp_in_262144"
        / "tokens_4.pt"
    ).is_file()
    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [21, 22, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["deduplicate"] is True
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_can_preserve_duplicate_rows_from_pretokenized_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [11, 12, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ],
            "attention_mask": [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_9"),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        deduplicate_shared_prompt_tokens=False,
        strict_shared_prompt_count=True,
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [11, 12, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["tensor_shape"] == [4, 4]
    assert metadata["unique_rows"] == 3
    assert metadata["deduplicate"] is False
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_uses_model_tokenizer_pad_token_when_metadata_omits_top_level_pad_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [11, 12, 0, 0],
                [21, 22, 0, 0],
                [31, 32, 33, 34],
                [41, 42, 43, 44],
            ]
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)
    (pretokenized_path / "sae_lens.json").write_text(
        json.dumps({"custom": {"pad_token_id": 0}}),
        encoding="utf-8",
    )

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    output_dir = tmp_path / "layer_9"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(output_dir),
        n_prompts_total=4,
        n_tokens_in_prompt=4,
        n_prompts_in_forward_pass=2,
        primary_acts_batch_size=1,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        auto_prompt_bucket_schedule=True,
        prompt_bucket_ceilings=(2,),
        prompt_bucket_scale_limit=2.0,
        prompt_primary_acts_scale_limit=2.0,
        prompt_batch_size_round_to=2,
    )
    runner.model = cast(Any, SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    runner.model_id = "google/gemma-3-1b-it"
    runner.hook_name = "blocks.9.hook_mlp_in"
    runner.sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=262144))

    runner._setup_output_directory()
    tokens = runner.get_tokens()
    schedule = runner._load_prompt_bucket_schedule(tokens)

    metadata = json.loads(
        (pretokenized_path / "tokens_4.metadata.json").read_text(encoding="utf-8")
    )

    assert tokens.tolist() == [
        [11, 12, 0, 0],
        [21, 22, 0, 0],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    assert metadata["pad_token_id"] == 0
    assert schedule == [
        {"prompt_indices": [0, 1], "seq_length": 2, "primary_acts_batch_size": 2},
        {"prompt_indices": [2, 3], "seq_length": 4, "primary_acts_batch_size": 1},
    ]


def test_auto_bucket_schedule_strict_count_errors_on_deduped_shortfall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[11, 12, 0, 0], [11, 12, 0, 0], [31, 32, 33, 34]],
            "attention_mask": [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
        }
    )
    pretokenized_path = tmp_path / "rte_tokens"
    pretokenized_path.mkdir(parents=True)

    def fake_load_from_disk(path: str) -> Dataset:
        assert path == str(pretokenized_path)
        return dataset

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.load_from_disk",
        fake_load_from_disk,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gemma-scope-2-1b-it-transcoders-all",
        sae_path="layer_9_width_262k_l0_small_affine",
        outputs_dir=str(tmp_path / "layer_9"),
        n_prompts_total=3,
        n_tokens_in_prompt=4,
        huggingface_dataset_path="aps/super_glue",
        pretokenized_dataset_path=str(pretokenized_path),
        strict_shared_prompt_count=True,
        auto_prompt_bucket_schedule=True,
    )

    with pytest.raises(ValueError, match="did not satisfy the requested prompt count"):
        runner._prepare_shared_tokens_from_pretokenized_dataset(
            pretokenized_path / "tokens_3.pt"
        )


def test_generate_tokens_requests_cpu_batches() -> None:
    class FakeActivationsStore:
        store_batch_size_prompts = 2

        def __init__(self) -> None:
            self.move_to_model_device_args: list[bool] = []

        def get_batch_tokens(self, move_to_model_device: bool = True):
            self.move_to_model_device_args.append(move_to_model_device)
            return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        n_prompts_total=2,
    )
    runner.cfg.shuffle_tokens = False

    fake_store = FakeActivationsStore()
    tokens = runner.generate_tokens(fake_store, n_prompts=2)

    assert fake_store.move_to_model_device_args == [False]
    assert tokens.device.type == "cpu"
    assert tokens.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_write_converter_input_artifact(tmp_path: Path) -> None:
    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    artifact_dir = tmp_path / "converter_inputs"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir="test_outputs",
        converter_input_artifact_dir=str(artifact_dir),
    )
    runner.vocab_dict = {1: "token1"}
    runner.model = SimpleNamespace(cfg=SimpleNamespace(d_vocab=50257))
    runner.model_id = "gpt2-small"
    runner.layer = 5
    runner.hook_name = "blocks.5.hook_resid_pre"

    feature_data = SimpleNamespace(feature_data_dict={1: {"feature_index": 1}})

    artifact_path = runner._write_converter_input_artifact(feature_data, batch_num=3)

    assert artifact_path == artifact_dir / "converter_input_batch_3.pt"
    assert artifact_path.is_file()
    snapshot = torch.load(artifact_path, weights_only=False)
    assert snapshot["feature_data_dict"] == feature_data.feature_data_dict
    assert snapshot["runner_cfg"].converter_input_artifact_dir == str(artifact_dir)
    assert snapshot["vocab_dict"] == {1: "token1"}
    assert snapshot["model_d_vocab"] == 50257
    assert snapshot["batch_num"] == 3


def test_run_feature_batch_with_optional_profile_writes_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_events: list[dict[str, object]] = []

    class _FakeSaeVisRunner:
        def __init__(self, cfg) -> None:
            self.cfg = cfg

        def run(self, encoder, model, tokens):
            assert encoder is runner.sae
            assert model is runner.model
            assert tokens.shape == (1, 2)
            return {"result": "profiled"}

    def _record(event: str, /, **fields: object) -> None:
        perf_events.append({"event": event, **fields})

    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.SaeVisRunner",
        _FakeSaeVisRunner,
    )
    monkeypatch.setattr(
        "sae_dashboard.neuronpedia.neuronpedia_runner.log_perf_event",
        _record,
    )

    runner = NeuronpediaRunner.__new__(NeuronpediaRunner)
    trace_dir = tmp_path / "torch_profiles"
    runner.cfg = NeuronpediaRunnerConfig(
        sae_set="gpt2-small-res-jb",
        sae_path="blocks.5.hook_resid_pre",
        outputs_dir=str(tmp_path),
        torch_profile=True,
        torch_profile_dir=str(trace_dir),
    )
    runner.sae = SimpleNamespace()
    runner.model = SimpleNamespace()

    result = runner._run_feature_batch_with_optional_profile(
        feature_vis_config_gpt=SimpleNamespace(),
        tokens=torch.ones(1, 2, dtype=torch.long),
        feature_batch_count=4,
    )

    assert result == {"result": "profiled"}
    trace_event = next(
        event for event in perf_events if event.get("event") == "torch_profile_trace"
    )
    trace_path = Path(str(trace_event["path"]))
    assert trace_path == trace_dir / "batch-4.trace.json"
    assert trace_path.is_file()


# def test_add_prefix_suffix_to_tokens(neuronpedia_runner: NeuronpediaRunner) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 1:4].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )


# def test_add_prefix_suffix_to_tokens_prepend_bos_false(
#     neuronpedia_runner: NeuronpediaRunner,
# ) -> None:
#     # modify the config to add a prefix / suffix
#     neuronpedia_runner.cfg.prefix_tokens = [101, 102, 103]  # Example prefix tokens
#     neuronpedia_runner.cfg.suffix_tokens = [104, 105, 106]  # Example suffix tokens

#     # get the tokens
#     neuronpedia_runner.sae.cfg.prepend_bos = False
#     tokens = neuronpedia_runner.get_tokens()
#     tokens = neuronpedia_runner.add_prefix_suffix_to_tokens(tokens)

#     # check that the tokens have the prefix and suffix
#     assert torch.allclose(tokens[:, 0:3].cpu(), torch.tensor([101, 102, 103]))
#     assert torch.allclose(tokens[:, -3:].cpu(), torch.tensor([104, 105, 106]))

#     # assert the first token position is still the bos
#     assert not torch.allclose(
#         tokens[:, 0].cpu(),
#         torch.tensor(
#             [neuronpedia_runner.model.to_single_token("<|endoftext|>")],
#             dtype=torch.int64,
#         ),
#     )
