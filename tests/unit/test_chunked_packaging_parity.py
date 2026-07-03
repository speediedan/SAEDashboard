# pyright: reportMissingTypeStubs=false
"""Parity tests for the chunked-device feature-statistics / activation-histogram
builders (Phase 6.2 slice 2) and the memoized activation-row detokenization."""

from typing import Any

import pytest
import torch

from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import FeatureStatistics, HistogramData
from tests.helpers import build_sae_vis_cfg


def _stats_fixture(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    flat_data = (torch.randn(9, 40) * torch.linspace(0.2, 3.0, 9).unsqueeze(-1)).to(
        dtype
    )
    flat_data[3] = 0.0  # all-zero row
    flat_data[4] = -flat_data[4].abs()  # all-negative row
    flat_data[5] = 0.75  # constant-positive row
    valid_mask = torch.ones(40, dtype=torch.bool)
    valid_mask[::7] = False
    return flat_data, valid_mask


def _reference_scalar_table(flat_data: torch.Tensor, valid_mask: torch.Tensor) -> Any:
    feature_stats_input = flat_data[:, valid_mask]
    if feature_stats_input.shape[-1] == 0:
        feature_stats_input = torch.zeros(
            (flat_data.shape[0], 1),
            dtype=flat_data.dtype,
            device=flat_data.device,
        )
    return FeatureStatistics.create_arrow_table(
        data=feature_stats_input,
        include_quantiles=False,
    )


def _reference_histogram_table(
    flat_data: torch.Tensor,
    valid_mask: torch.Tensor,
    titles: list[str],
    n_bins: int = 10,
) -> Any:
    masked = flat_data * valid_mask.to(device=flat_data.device, dtype=flat_data.dtype)
    return HistogramData.from_data_batch_arrow_table(
        data=masked,
        n_bins=n_bins,
        tickmode="5 ticks",
        title=None,
        positive_only=True,
        titles=titles,
        backend="torch",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scalar_stats_from_flat_valid_matches_gathered(dtype: torch.dtype) -> None:
    flat_data, valid_mask = _stats_fixture(dtype)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        valid_indices,
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(flat_data, valid_mask)

    assert chunked.schema.names == reference.schema.names
    assert chunked.to_pydict() == reference.to_pydict()


def test_scalar_stats_from_flat_valid_empty_valid_matches_zero_substitution() -> None:
    flat_data, _ = _stats_fixture()
    empty_indices = torch.zeros(0, dtype=torch.long)

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        empty_indices,
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(
        flat_data, torch.zeros(flat_data.shape[-1], dtype=torch.bool)
    )

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scalar_stats_from_flat_valid_cuda_matches_cpu_reference() -> None:
    flat_data, valid_mask = _stats_fixture()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

    chunked = FeatureStatistics.create_scalar_arrow_table_from_flat_valid(
        flat_data,
        valid_indices,
        compute_device="cuda",
        row_chunk_size=4,
    )
    reference = _reference_scalar_table(flat_data, valid_mask)

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_histograms_from_flat_valid_match_masked_reference(dtype: torch.dtype) -> None:
    flat_data, valid_mask = _stats_fixture(dtype)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        valid_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        row_chunk_size=2,
        row_batch_size=3,
    )
    reference = _reference_histogram_table(flat_data, valid_mask, titles)

    assert chunked.schema.names == reference.schema.names
    assert chunked.to_pydict() == reference.to_pydict()


def test_histograms_from_flat_valid_empty_valid_produces_default_rows() -> None:
    flat_data, _ = _stats_fixture()
    empty_indices = torch.zeros(0, dtype=torch.long)
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        empty_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        row_chunk_size=4,
    )
    reference = _reference_histogram_table(
        flat_data, torch.zeros(flat_data.shape[-1], dtype=torch.bool), titles
    )

    assert chunked.to_pydict() == reference.to_pydict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_histograms_from_flat_valid_cuda_matches_cpu_reference() -> None:
    flat_data, valid_mask = _stats_fixture()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    titles = [f"row {index}" for index in range(flat_data.shape[0])]

    chunked = HistogramData.from_flat_valid_data_batch_arrow_table(
        flat_data,
        valid_indices,
        n_bins=10,
        tickmode="5 ticks",
        titles=titles,
        compute_device="cuda",
        row_chunk_size=2,
    ).to_pydict()
    reference = _reference_histogram_table(flat_data, valid_mask, titles).to_pydict()

    # Bin contents, edges semantics, ticks, and titles must match exactly; the
    # display-only bin midpoints (`bar_values`) may differ in the last rounded
    # decimal across devices (float32 mul-add rounding at the 5-decimal boundary).
    for column in ("row_index", "bar_heights", "tick_vals", "title"):
        assert chunked[column] == reference[column], column
    for row_chunked, row_reference in zip(
        chunked["bar_values"], reference["bar_values"]
    ):
        assert len(row_chunked) == len(row_reference)
        for value_chunked, value_reference in zip(row_chunked, row_reference):
            # one 5-decimal display-rounding step (plus float representation slack)
            assert abs(value_chunked - value_reference) <= 1.05e-5


class _CountingTokenizer:
    def __init__(self) -> None:
        self.ids_decoded = 0

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        self.ids_decoded += len(token_ids)
        return [f"tok{token_id}" for token_id in token_ids]


class _FakeModel:
    def __init__(self) -> None:
        self.tokenizer = _CountingTokenizer()


def test_decode_token_ids_cached_matches_direct_and_dedupes() -> None:
    runner = SaeVisRunner(build_sae_vis_cfg())
    model: Any = _FakeModel()

    first = runner._decode_token_ids_cached(model, [5, 3, 5, 9])
    assert first == ["tok5", "tok3", "tok5", "tok9"]
    assert model.tokenizer.ids_decoded == 3  # 5, 3, 9 decoded once

    second = runner._decode_token_ids_cached(model, [9, 9, 5, 7])
    assert second == ["tok9", "tok9", "tok5", "tok7"]
    assert model.tokenizer.ids_decoded == 4  # only 7 was new

    assert runner._decode_token_ids_cached(model, []) == []
    assert model.tokenizer.ids_decoded == 4
