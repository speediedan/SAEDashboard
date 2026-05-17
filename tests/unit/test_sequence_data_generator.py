import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer

from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceDataGenerator,
)
from tests.helpers import build_sae_vis_cfg


@pytest.fixture
def sequence_data_generator(
    model: HookedTransformer, tokens: torch.Tensor
) -> SequenceDataGenerator:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    return SequenceDataGenerator(cfg, tokens, model.W_U)


def test_get_sequences_data_expected_duplicates(
    sequence_data_generator: SequenceDataGenerator,
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> None:
    token_scores = torch.arange(tokens.numel(), dtype=torch.float32).reshape(
        tokens.shape
    )
    feat_acts = token_scores / max(1, token_scores.numel())
    feat_logits = torch.linspace(-1.0, 1.0, model.cfg.d_vocab, dtype=torch.float32)
    resid_post = token_scores.unsqueeze(-1).repeat(1, 1, model.cfg.d_model)
    feature_resid_dir = torch.linspace(0.1, 1.0, model.cfg.d_model, dtype=torch.float32)

    sequence_multi_group_data = sequence_data_generator.get_sequences_data(
        feat_acts, feat_logits, resid_post, feature_resid_dir
    )

    all_sequence_data = []
    group_sequence_pairs = []
    for i, group in enumerate(sequence_multi_group_data.seq_group_data):
        all_sequence_data.extend(group.seq_data)
        group_sequence_pairs.extend(
            [(i, sd.original_index, sd.qualifying_token_index) for sd in group.seq_data]
        )

    # Count occurrences of each (original_index, qualifying_token_index) pair
    from collections import Counter

    pair_counts = Counter(
        (sd.original_index, sd.qualifying_token_index) for sd in all_sequence_data
    )

    # Check for duplicates within the same group
    duplicates_in_same_group = False
    for i, group in enumerate(sequence_multi_group_data.seq_group_data):
        group_pairs = [
            (sd.original_index, sd.qualifying_token_index) for sd in group.seq_data
        ]
        group_pair_counts = Counter(group_pairs)
        if any(count > 1 for count in group_pair_counts.values()):
            duplicates_in_same_group = True

    # Assertions
    assert not duplicates_in_same_group, "Duplicates found within the same group"
    assert len(sequence_multi_group_data.seq_group_data[0].seq_data) == 20, (
        "TOP ACTIVATIONS group should have 20 sequences"
    )

    # Check that duplicates only occur between TOP ACTIVATIONS and one other group
    for pair, count in pair_counts.items():
        if count > 1:
            groups_with_pair = [
                i for i, oi, qti in group_sequence_pairs if (oi, qti) == pair
            ]
            assert 0 in groups_with_pair, (
                f"Duplicate {pair} not in TOP ACTIVATIONS group"
            )
            assert len(groups_with_pair) == 2, (
                f"Duplicate {pair} found in more than two groups: {groups_with_pair}"
            )


def test_package_sequences_data_no_duplicates(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.randint(0, 1000, (10, 5))
    feat_acts_coloring = torch.randn(10, 5)
    feat_logits = torch.randn(1000)
    indices_dict = {
        "Group1": torch.tensor([[0, 1], [1, 2]]),
        "Group2": torch.tensor([[2, 3], [3, 4]]),
    }
    indices_bold = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    all_sequence_data = []
    for group in (
        sequence_multi_group_data.seq_group_data
    ):  # Changed from sequence_groups to seq_group_data
        all_sequence_data.extend(group.seq_data)  # Changed from sequences to seq_data

    assert len(all_sequence_data) == len(
        set((sd.original_index, sd.qualifying_token_index) for sd in all_sequence_data)
    )


def test_package_sequences_data_rounds_feature_acts_to_4_decimals(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3]])
    feat_acts_coloring = torch.tensor([[0.12344, 0.12345, -1.23456]])
    feat_logits = torch.randn(10)
    indices_dict = {"Group1": torch.tensor([[0, 1]])}
    indices_bold = torch.tensor([[0, 1]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    seq_data = sequence_multi_group_data.seq_group_data[0].seq_data[0]

    assert seq_data.feat_acts == [
        round(float(value), 4) for value in feat_acts_coloring[0].tolist()
    ]


def test_package_sequences_data_rounds_bfloat16_feature_acts(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3]])
    feat_acts_coloring = torch.tensor(
        [[0.12344, 0.12345, -1.23456]], dtype=torch.bfloat16
    )
    feat_logits = torch.randn(10)
    indices_dict = {"Group1": torch.tensor([[0, 1]])}
    indices_bold = torch.tensor([[0, 1]])

    sequence_multi_group_data = sequence_data_generator.package_sequences_data(
        token_ids, feat_acts_coloring, feat_logits, indices_dict, indices_bold
    )

    seq_data = sequence_multi_group_data.seq_group_data[0].seq_data[0]

    assert seq_data.feat_acts == [
        round(float(value), 4) for value in feat_acts_coloring[0].float().tolist()
    ]


def test_build_sequence_coordinate_table_round_trips_to_nested_sequence_data(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
    )

    assert coordinate_table.group_names == ["Group1", "Group2"]
    assert coordinate_table.group_sizes == [2, 1]
    assert coordinate_table.group_offsets == [0, 2, 3]
    assert coordinate_table.original_indices.tolist() == [10, 11, 12]
    assert coordinate_table.qualifying_token_indices.tolist() == [1, 2, 0]
    assert coordinate_table.source_token_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert coordinate_table.token_ids.tolist() == token_ids.tolist()
    assert (
        coordinate_table.feat_acts.tolist()
        == np.around(
            feat_acts_coloring.numpy().astype(np.float64, copy=False), 4
        ).tolist()
    )
    assert coordinate_table.token_logits.tolist() == feat_logits[token_ids].tolist()

    sequence_multi_group_data = coordinate_table.to_sequence_multi_group_data()

    assert [group.title for group in sequence_multi_group_data.seq_group_data] == [
        "Group1",
        "Group2",
    ]
    assert [
        len(group.seq_data) for group in sequence_multi_group_data.seq_group_data
    ] == [
        2,
        1,
    ]
    first_sequence = sequence_multi_group_data.seq_group_data[0].seq_data[0]
    assert first_sequence.original_index == 10
    assert first_sequence.qualifying_token_index == 1
    assert first_sequence.token_ids == [1, 2, 3]
    assert first_sequence.feat_acts == [0.1234, 0.1235, -1.2346]
    assert first_sequence.token_logits == feat_logits[token_ids[0]].tolist()


def test_sequence_coordinate_table_writes_flat_arrow_and_parquet_rows(
    sequence_data_generator: SequenceDataGenerator,
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_ipc = pytest.importorskip("pyarrow.ipc")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    datasets = pytest.importorskip("datasets")

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])
    source_token_indices = torch.tensor([[7, 8, 9], [10, 11, 12], [20, 21, 22]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
        source_token_indices=source_token_indices,
    )

    row_table = coordinate_table.to_sequence_row_arrow_table(feature_index=42)
    records = row_table.to_pylist()

    assert row_table.num_rows == 9
    assert row_table.schema.names == [
        "feature_index",
        "sequence_index",
        "group_index",
        "group_name",
        "group_sequence_index",
        "context_token_index",
        "original_index",
        "qualifying_token_index",
        "source_token_index",
        "token_id",
        "feat_act",
        "token_logit",
    ]
    assert records[0]["feature_index"] == 42
    assert records[0]["sequence_index"] == 0
    assert records[0]["group_index"] == 0
    assert records[0]["group_name"] == "Group1"
    assert records[0]["group_sequence_index"] == 0
    assert records[0]["context_token_index"] == 0
    assert records[0]["original_index"] == 10
    assert records[0]["qualifying_token_index"] == 1
    assert records[0]["source_token_index"] == 7
    assert records[0]["token_id"] == 1
    assert records[0]["feat_act"] == pytest.approx(0.1234)
    assert records[0]["token_logit"] == pytest.approx(float(feat_logits[1]))
    assert records[3]["group_sequence_index"] == 1
    assert records[-1]["group_name"] == "Group2"
    assert records[-1]["source_token_index"] == 22

    arrow_path = tmp_path / "sequence_rows.arrow"
    parquet_path = tmp_path / "sequence_rows.parquet"

    assert coordinate_table.write_sequence_row_arrow_ipc(
        arrow_path, feature_index=42
    ) == len(records)
    assert coordinate_table.write_sequence_row_parquet(
        parquet_path, feature_index=42
    ) == len(records)

    with pyarrow.memory_map(str(arrow_path), "r") as source:
        arrow_records = pyarrow_ipc.open_file(source).read_all().to_pylist()
    parquet_records = pyarrow_parquet.read_table(str(parquet_path)).to_pylist()
    parquet_dataset = datasets.Dataset.from_parquet(str(parquet_path))

    assert arrow_records == records
    assert parquet_records == records
    assert parquet_dataset.num_rows == len(records)
    assert parquet_dataset[0]["feature_index"] == 42


def test_activation_copy_row_record_batch_from_activation_row_record_batch_matches_columns(
    sequence_data_generator: SequenceDataGenerator,
) -> None:
    pytest.importorskip("pyarrow")

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [6, 5, 4]])
    feat_acts_coloring = torch.tensor(
        [
            [0.12344, 0.12345, -1.23456],
            [2.0, 0.0, -0.00004],
            [3.14159, -2.71828, 1.41421],
        ]
    )
    feat_logits = torch.linspace(-1.0, 1.0, 10)
    indices_dict = {
        "Group1": torch.tensor([[10, 1], [11, 2]]),
        "Group2": torch.tensor([[12, 0]]),
    }
    indices_bold = torch.tensor([[10, 1], [11, 2], [12, 0]])

    coordinate_table = sequence_data_generator.build_sequence_coordinate_table(
        token_ids=token_ids,
        feat_acts_coloring=feat_acts_coloring,
        feat_logits=feat_logits,
        indices_dict=indices_dict,
        indices_bold=indices_bold,
    )
    activation_columns = coordinate_table.to_activation_row_columns(
        feature_index=42,
        decode_token_ids=lambda token_ids: [f"tok-{token_id}" for token_id in token_ids],
    )
    activation_record_batch = SequenceCoordinateTable.activation_row_arrow_record_batch_from_columns(
        activation_columns
    )

    converted_from_batch = (
        SequenceCoordinateTable.activation_copy_row_arrow_record_batch_from_activation_row_record_batch(
            activation_record_batch,
            model_id="test-model",
            layer="9-test-source",
            creator_id="test-creator",
            created_at="2026-01-02T03:04:05",
            activation_id_prefix="test-activation",
        )
    )
    converted_from_columns = SequenceCoordinateTable.activation_copy_row_arrow_record_batch_from_activation_columns(
        activation_columns,
        model_id="test-model",
        layer="9-test-source",
        creator_id="test-creator",
        created_at="2026-01-02T03:04:05",
        activation_id_prefix="test-activation",
    )

    assert converted_from_batch.schema == converted_from_columns.schema
    assert converted_from_batch.to_pydict() == converted_from_columns.to_pydict()


def test_get_sequence_coordinate_table_matches_get_sequences_data(
    sequence_data_generator: SequenceDataGenerator,
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> None:
    token_scores = torch.arange(tokens.numel(), dtype=torch.float32).reshape(
        tokens.shape
    )
    feat_acts = token_scores / max(1, token_scores.numel())
    feat_logits = torch.linspace(-1.0, 1.0, model.cfg.d_vocab, dtype=torch.float32)
    resid_post = token_scores.unsqueeze(-1).repeat(1, 1, model.cfg.d_model)
    feature_resid_dir = torch.linspace(0.1, 1.0, model.cfg.d_model, dtype=torch.float32)

    coordinate_table = sequence_data_generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
    )
    nested_from_table = coordinate_table.to_sequence_multi_group_data()
    nested_direct = sequence_data_generator.get_sequences_data(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
    )

    assert nested_from_table == nested_direct


def test_get_sequence_coordinate_table_lazy_gpu_selection_matches_eager_cpu() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 0  # type: ignore

    tokens = torch.tensor(
        [
            [11, 12, 13, 14, 0],
            [21, 22, 23, 0, 0],
        ],
        dtype=torch.long,
    )
    selection_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )
    generator = SequenceDataGenerator(cfg, tokens, torch.zeros(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.8, 0.1, 1.2, 9.0],
            [0.0, 0.6, 1.0, 8.0, 7.0],
        ],
        dtype=torch.float32,
    )
    feat_logits = torch.linspace(-1.0, 1.0, 32)
    resid_post = torch.empty(0)
    feature_resid_dir = torch.empty(0)

    eager_table = generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
        selection_backend="eager_cpu",
    )
    lazy_table = generator.get_sequence_coordinate_table(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
        selection_backend="lazy_gpu",
    )

    assert (
        lazy_table.to_sequence_multi_group_data()
        == eager_table.to_sequence_multi_group_data()
    )


def test_get_sequences_data_selection_mask_excludes_ignored_padding_positions() -> None:
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 5  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 0  # type: ignore

    tokens = torch.tensor(
        [
            [11, 12, 13, 0, 0, 0],
            [21, 22, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    feat_logits = torch.randn(32)
    resid_post = torch.zeros(tokens.shape[0], tokens.shape[1], 4)
    feature_resid_dir = torch.randn(4)
    selection_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, False, False, False, False],
        ]
    )

    sequence_multi_group_data = generator.get_sequences_data(
        feat_acts=feat_acts,
        feat_logits=feat_logits,
        resid_post=resid_post,
        feature_resid_dir=feature_resid_dir,
        selection_mask=selection_mask,
    )

    top_group = sequence_multi_group_data.seq_group_data[0].seq_data
    assert len(top_group) == 5
    assert all(
        selection_mask[seq.original_index, seq.qualifying_token_index].item()
        for seq in top_group
    )


def test_get_indices_dict_lazy_gpu_matches_eager_cpu_indices_with_selection_mask() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 4  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(18, dtype=torch.long).reshape(3, 6)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0, 0.0],
            [0.0, 1.25, 1.5, 1.75, 2.0, 0.0],
            [0.0, 0.4, 0.8, 1.2, 1.6, 0.0],
        ],
        dtype=torch.float32,
    )
    selection_mask = torch.tensor(
        [
            [False, True, True, True, True, False],
            [False, True, True, True, True, False],
            [False, True, True, True, True, False],
        ]
    )

    random.seed(12345)
    eager_indices_dict, eager_indices_bold, eager_n_bold = (
        generator.get_indices_dict_eager_cpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )
    random.seed(12345)
    lazy_indices_dict, lazy_indices_bold, lazy_n_bold = (
        generator.get_indices_dict_lazy_gpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )

    assert list(lazy_indices_dict) == list(eager_indices_dict)
    assert lazy_n_bold == eager_n_bold
    assert torch.equal(lazy_indices_bold, eager_indices_bold)
    for group_name, eager_indices in eager_indices_dict.items():
        assert torch.equal(lazy_indices_dict[group_name], eager_indices)


def test_get_indices_dict_lazy_gpu_matches_eager_cpu_for_bfloat16_interval_boundaries() -> (
    None
):
    cfg: SaeVisConfig = build_sae_vis_cfg()
    cfg.feature_centric_layout.seq_cfg.buffer = None  # type: ignore
    cfg.feature_centric_layout.seq_cfg.top_acts_group_size = 1  # type: ignore
    cfg.feature_centric_layout.seq_cfg.n_quantiles = 5  # type: ignore
    cfg.feature_centric_layout.seq_cfg.quantile_group_size = 16  # type: ignore

    tokens = torch.arange(8, dtype=torch.long).reshape(1, 8)
    generator = SequenceDataGenerator(cfg, tokens, torch.randn(4, 32))
    feat_acts = torch.tensor(
        [[0.0, 44.25, 90.0, 121.5, 133.0, 177.0, 221.0, 0.0]],
        dtype=torch.bfloat16,
    )
    selection_mask = torch.tensor(
        [[True, True, True, True, True, True, True, False]],
        dtype=torch.bool,
    )

    random.seed(12345)
    eager_indices_dict, eager_indices_bold, eager_n_bold = (
        generator.get_indices_dict_eager_cpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )
    random.seed(12345)
    lazy_indices_dict, lazy_indices_bold, lazy_n_bold = (
        generator.get_indices_dict_lazy_gpu(
            generator.buffer,
            feat_acts,
            selection_mask=selection_mask,
        )
    )

    assert list(lazy_indices_dict) == list(eager_indices_dict)
    assert lazy_n_bold == eager_n_bold
    assert torch.equal(lazy_indices_bold, eager_indices_bold)
    for group_name, eager_indices in eager_indices_dict.items():
        assert torch.equal(lazy_indices_dict[group_name], eager_indices)
