from pathlib import Path
from types import SimpleNamespace

import torch

import sae_dashboard.perf_logging as perf_logging
import sae_dashboard.sae_vis_runner as sae_vis_runner_module
from sae_dashboard.components import FeatureTablesData, LogitsTableData
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


class _FakeFeatureDataGenerator:
    def get_feature_data(self, features, progress):
        del features, progress
        all_feat_acts = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)
        feature_resid_dir = torch.tensor([[1.0]], dtype=torch.float32)
        feature_out_dir = torch.tensor([[1.0]], dtype=torch.float32)
        return all_feat_acts, None, feature_resid_dir, feature_out_dir, None, None, None


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
    ):
        del feat_acts, feat_logits, resid_post, feature_resid_dir, selection_mask
        return []


class _FakeEncoder:
    def __init__(self) -> None:
        self.hook_z_reshaping_mode = False

    def fold_W_dec_norm(self) -> None:
        return None


def _capture_perf_events(monkeypatch) -> list[dict[str, object]]:
    perf_events: list[dict[str, object]] = []

    def _record(event: str, /, **fields: object) -> None:
        perf_events.append({"event": event, **fields})

    monkeypatch.setattr(perf_logging, "log_perf_event", _record)
    monkeypatch.setattr(sae_vis_runner_module, "log_perf_event", _record)
    return perf_events


def test_SaeVisRunner_cpu_eager_profiling_surfaces_stage_timings_and_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    perf_events = _capture_perf_events(monkeypatch)
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

    cfg = SaeVisConfig(
        hook_point="blocks.0.hook_resid_pre",
        features=[0],
        minibatch_size_features=1,
        minibatch_size_tokens=1,
        quantile_feature_batch_size=1,
        device="cpu",
        dtype="float32",
        ignore_tokens={0},
        log_performance=True,
        cleanup_each_minibatch=True,
        sequence_replay_artifact_dir=tmp_path / "sequence_replay_artifacts",
    )
    tokens = torch.tensor([[7, 0]], dtype=torch.long)

    sae_vis_data = SaeVisRunner(cfg).run(
        encoder=_FakeEncoder(),
        model=SimpleNamespace(W_U=torch.tensor([[1.0]], dtype=torch.float32)),
        tokens=tokens,
    )

    stage_names = {
        str(event["stage"])
        for event in perf_events
        if event.get("event") == "stage_timing"
    }
    assert {
        "activation_and_encode_total",
        "logits_projection",
        "feature_statistics_packaging",
        "logits_histogram_packaging",
        "activation_histogram_packaging",
        "feature_table_packaging",
        "logits_table_packaging",
        "sequence_packaging",
    }.issubset(stage_names)

    packaging_summary = next(
        event for event in perf_events if event.get("event") == "packaging_shape_summary"
    )
    assert packaging_summary["valid_token_count"] == 1
    assert packaging_summary["token_shape"] == [1, 2]

    artifact_event = next(
        event for event in perf_events if event.get("event") == "sequence_replay_artifact"
    )
    artifact_path = Path(str(artifact_event["path"]))
    assert artifact_path.is_file()

    artifact_payload = torch.load(artifact_path, weights_only=False)
    assert artifact_payload["feature_indices"] == [0]
    assert artifact_payload["token_shape"] == [1, 2]
    assert artifact_payload["valid_token_count"] == 1

    assert sae_vis_data.feature_stats.max == [1.0]
    assert sae_vis_data.feature_stats.frac_nonzero == [1.0]
    assert (
        sae_vis_data.feature_data_dict[0].acts_histogram_data.title
        == "ACTIVATIONS<br>DENSITY = 100.000%"
    )
