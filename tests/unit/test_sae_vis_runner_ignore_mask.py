from types import SimpleNamespace

import torch

from sae_dashboard.components import FeatureTablesData, LogitsTableData
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


class _FakeFeatureDataGenerator:
    def get_feature_data(self, features, progress):
        all_feat_acts = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)
        feature_resid_dir = torch.tensor([[1.0]], dtype=torch.float32)
        return all_feat_acts, None, feature_resid_dir, None, None, None, None


class _FakeSequenceDataGenerator:
    def __init__(self, cfg, tokens, W_U):
        self.cfg = cfg
        self.tokens = tokens
        self.W_U = W_U

    def get_sequences_data(
        self,
        feat_acts,
        feat_logits,
        resid_post,
        feature_resid_dir,
        selection_mask=None,
        selection_backend=None,
    ):
        return []


class _FakeEncoder:
    def __init__(self) -> None:
        self.hook_z_reshaping_mode = False

    def fold_W_dec_norm(self) -> None:
        return None


def test_feature_statistics_ignore_mask_excludes_ignored_tokens(monkeypatch) -> None:
    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
    )
    tokens = torch.tensor([[7, 0]], dtype=torch.long)

    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.FeatureDataGeneratorFactory.create",
        lambda cfg, model, encoder, tokens: _FakeFeatureDataGenerator(),
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.SequenceDataGenerator",
        _FakeSequenceDataGenerator,
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_features_table_data",
        lambda **kwargs: {
            name: [value] for name, value in FeatureTablesData().__dict__.items()
        },
    )
    monkeypatch.setattr(
        "sae_dashboard.sae_vis_runner.get_logits_table_data",
        lambda **kwargs: LogitsTableData(),
    )

    fake_model = SimpleNamespace(W_U=torch.tensor([[1.0]], dtype=torch.float32))
    fake_encoder = _FakeEncoder()

    sae_vis_data = SaeVisRunner(cfg).run(
        encoder=fake_encoder, model=fake_model, tokens=tokens
    )

    assert sae_vis_data.feature_stats.max == [1.0]
    assert sae_vis_data.feature_stats.frac_nonzero == [1.0]
    assert (
        sae_vis_data.feature_data_dict[0].acts_histogram_data.title
        == "ACTIVATIONS<br>DENSITY = 100.000%"
    )
