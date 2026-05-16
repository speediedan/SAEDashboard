import pytest
import torch

from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.sae_vis_data import SaeVisConfig


def test_batch_tokens_uses_prompt_minibatch_schedule() -> None:
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 0],
            [10, 11, 12, 0, 0, 0],
            [13, 14, 15, 16, 17, 18],
        ],
        dtype=torch.long,
    )
    generator = FeatureDataGenerator.__new__(FeatureDataGenerator)
    generator.cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        prompt_minibatch_schedule=[
            {
                "prompt_indices": [2, 3],
                "seq_length": 4,
                "primary_acts_batch_size": 1,
            },
            {
                "prompt_indices": [0, 1],
                "seq_length": 5,
                "primary_acts_batch_size": 2,
            },
        ],
    )

    minibatches = FeatureDataGenerator.batch_tokens(generator, tokens)

    assert [minibatch.prompt_indices for minibatch in minibatches] == [(2, 3), (0, 1)]
    assert [tuple(minibatch.tokens.shape) for minibatch in minibatches] == [(2, 4), (2, 5)]
    assert [minibatch.primary_acts_batch_size for minibatch in minibatches] == [1, 2]
    assert minibatches[0].tokens.tolist() == [[10, 11, 12, 0], [13, 14, 15, 16]]
    assert minibatches[1].tokens.tolist() == [[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]]


def test_scatter_feature_act_chunk_restores_original_prompt_order() -> None:
    destination = torch.zeros((4, 6, 2), dtype=torch.bfloat16)
    shorter_bucket_chunk = torch.full((2, 6, 2), 1, dtype=torch.bfloat16)
    longer_bucket_chunk = torch.full((2, 6, 2), 2, dtype=torch.bfloat16)

    FeatureDataGenerator._scatter_feature_act_chunk(
        destination,
        shorter_bucket_chunk,
        prompt_indices=(2, 3),
    )
    FeatureDataGenerator._scatter_feature_act_chunk(
        destination,
        longer_bucket_chunk,
        prompt_indices=(0, 1),
    )

    assert destination[0].eq(2).all()
    assert destination[1].eq(2).all()
    assert destination[2].eq(1).all()
    assert destination[3].eq(1).all()


def test_pad_sequence_tensor_zero_fills_trimmed_tail() -> None:
    sequence_tensor = torch.tensor(
        [[[1.0], [2.0], [3.0]]],
        dtype=torch.float32,
    )

    padded = FeatureDataGenerator._pad_sequence_tensor(
        sequence_tensor,
        target_seq_len=5,
    )

    assert tuple(padded.shape) == (1, 5, 1)
    assert padded[:, :3].tolist() == sequence_tensor.tolist()
    assert padded[:, 3:].tolist() == [[[0.0], [0.0]]]
