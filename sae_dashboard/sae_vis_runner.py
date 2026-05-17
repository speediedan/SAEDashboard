import gc
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Union

import einops
import numpy as np
import torch
from jaxtyping import Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE, HookedSAETransformer
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_dashboard.components import (
    ActsHistogramData,
    DecoderWeightsDistribution,
    FeatureTablesData,
    LogitsHistogramData,
)
from sae_dashboard.data_parsing_fns import (
    get_features_table_data,
    get_logits_table_data,
)
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.huggingface_model_wrapper import (
    HFActivationConfig,
    HuggingFaceModelWrapper,
)
from sae_dashboard.perf_logging import log_perf_event, timed_stage
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.sequence_data_generator import SequenceDataGenerator
from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)
from sae_dashboard.utils_fns import FeatureStatistics


def _resolve_unembed_matrix(model: HookedSAETransformer) -> Tensor:
    if hasattr(model, "W_U"):
        return model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return model.unembed.W_U
    raise AttributeError(f"{type(model).__name__} does not expose W_U")


def _build_ignore_tokens_mask(
    cfg: SaeVisConfig,
    tokens: Int[Tensor, "batch seq"],
    target_device: torch.device | str,
) -> Tensor:
    ignore_tokens_mask = torch.ones_like(tokens, dtype=torch.bool)
    if cfg.ignore_tokens:
        ignore_tokens_mask &= ~torch.isin(
            tokens,
            torch.tensor(
                list(cfg.ignore_tokens),
                dtype=tokens.dtype,
                device=tokens.device,
            ),
        )
    if cfg.ignore_positions:
        ignore_positions_mask = torch.ones_like(tokens, dtype=torch.bool)
        ignore_positions_mask[:, cfg.ignore_positions] = False
        ignore_tokens_mask &= ignore_positions_mask
    return ignore_tokens_mask.to(target_device)


class FeatureDataGeneratorFactory:
    @staticmethod
    def create(
        cfg: SaeVisConfig,
        model: Union[HookedSAETransformer, AutoModelForCausalLM],
        encoder: SAE,  # type: ignore
        tokens: Int[Tensor, "batch seq"],
        tokenizer: AutoTokenizer = None,  # Required for HuggingFace models
    ) -> FeatureDataGenerator:
        """Builds a FeatureDataGenerator using the provided config and model.

        Args:
            cfg: The SaeVisConfig configuration
            model: Either a HookedSAETransformer (TransformerLens) or AutoModelForCausalLM (HuggingFace)
            encoder: The SAE encoder
            tokens: The input tokens
            tokenizer: Required when using HuggingFace models (cfg.use_huggingface=True)

        Returns:
            FeatureDataGenerator instance configured for the model type
        """
        if cfg.use_huggingface:
            # Use HuggingFace model wrapper
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when use_huggingface=True")

            # DFA is not yet supported for HuggingFace models
            if cfg.use_dfa:
                raise NotImplementedError(
                    "DFA (Direct Feature Attribution) is not yet supported for HuggingFace models. "
                    "Please use TransformerLens (use_huggingface=False) for DFA."
                )

            activation_config = HFActivationConfig(
                primary_hook_point=cfg.hook_point,
                auxiliary_hook_points=[],
            )
            wrapped_model = HuggingFaceModelWrapper(
                model=model,  # type: ignore
                tokenizer=tokenizer,
                activation_config=activation_config,
                dtype=DTYPES.get(cfg.dtype, torch.float32),
            )
        else:
            # Use TransformerLens model wrapper
            activation_config = ActivationConfig(
                primary_hook_point=cfg.hook_point,
                auxiliary_hook_points=(
                    [
                        re.sub(r"hook_z", "hook_v", cfg.hook_point),
                        re.sub(r"hook_z", "hook_pattern", cfg.hook_point),
                    ]
                    if cfg.use_dfa
                    else []
                ),
            )
            wrapped_model = TransformerLensWrapper(model, activation_config)  # type: ignore

        return FeatureDataGenerator(
            cfg=cfg,
            model=wrapped_model,
            encoder=encoder,
            tokens=tokens,  # type: ignore
        )


class SaeVisRunner:
    def __init__(self, cfg: SaeVisConfig) -> None:
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = DTYPES[self.cfg.dtype]
        if self.cfg.cache_dir is not None:
            self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.sequence_replay_artifact_dir is not None:
            self.cfg.sequence_replay_artifact_dir.mkdir(parents=True, exist_ok=True)

    def _write_sequence_replay_artifact(
        self,
        *,
        feature_batch_index: int,
        features: list[int],
        tokens: Int[Tensor, "batch seq"],
        selection_mask: Tensor,
        valid_token_count: int,
        all_feat_acts: Tensor,
        logits: Tensor,
        feature_resid_dir: Tensor,
    ) -> Path | None:
        if self.cfg.sequence_replay_artifact_dir is None:
            return None

        artifact_path = (
            self.cfg.sequence_replay_artifact_dir
            / f"feature-batch-{feature_batch_index:04d}.pt"
        )
        torch.save(
            {
                "feature_batch_index": feature_batch_index,
                "feature_indices": list(features),
                "token_shape": list(tokens.shape),
                "valid_token_count": valid_token_count,
                "tokens": tokens.detach().cpu(),
                "selection_mask": selection_mask.detach().cpu(),
                "feature_activations": all_feat_acts.detach().cpu(),
                "feature_logits": logits.detach().cpu(),
                "feature_resid_dir": feature_resid_dir.detach().cpu(),
            },
            artifact_path,
        )
        return artifact_path

    @torch.inference_mode()
    def run(
        self,
        encoder: SAE,  # type: ignore
        model: Union[HookedSAETransformer, AutoModelForCausalLM],
        tokens: Int[Tensor, "batch seq"],
        tokenizer: AutoTokenizer = None,  # Required for HuggingFace models
    ) -> SaeVisData:
        self.set_seeds()

        if "CLTLayerWrapper" in str(type(encoder)) or encoder.cfg.architecture() in [
            "temporal"
        ]:
            print("SaeVisRunner: Skipping fold_W_dec_norm() for CLT wrapper.")
        else:
            encoder.fold_W_dec_norm()

        if "CLTLayerWrapper" in str(type(encoder)):
            print("SaeVisRunner: Skipping hook_z_reshaping_mode check for CLT wrapper.")
        elif encoder.hook_z_reshaping_mode:
            encoder.turn_off_forward_pass_hook_z_reshaping()

        sae_vis_data = SaeVisData(cfg=self.cfg)
        time_logs = defaultdict(float)

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)
        progress = self.get_progress_bar(tokens, feature_batches, features_list)

        feature_data_generator = FeatureDataGeneratorFactory.create(
            self.cfg,
            model,
            encoder,
            tokens,
            tokenizer=tokenizer,
        )

        unembed_matrix = (
            self._get_hf_unembed_matrix(model)
            if self.cfg.use_huggingface
            else _resolve_unembed_matrix(model)
        )
        sequence_data_generator = SequenceDataGenerator(
            cfg=self.cfg,
            tokens=tokens,
            W_U=unembed_matrix,
        )

        all_consolidated_dfa_results = {feature_idx: {} for feature_idx in features_list}
        for feature_batch_index, features in enumerate(feature_batches):
            with timed_stage(
                self.cfg.log_performance,
                "activation_and_encode_total",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                (
                    all_feat_acts,
                    _,
                    feature_resid_dir,
                    feature_out_dir,
                    corrcoef_neurons,
                    corrcoef_encoder,
                    batch_dfa_results,
                ) = feature_data_generator.get_feature_data(features, progress)

            with timed_stage(
                self.cfg.log_performance,
                "logits_projection",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                logits = einops.einsum(
                    feature_resid_dir.to(
                        device=unembed_matrix.device,
                        dtype=unembed_matrix.dtype,
                    ),
                    unembed_matrix,
                    "feats d_model, d_model d_vocab -> feats d_vocab",
                ).to(self.device)

            ignore_tokens_mask = _build_ignore_tokens_mask(
                self.cfg,
                tokens,
                all_feat_acts.device,
            )
            flat_all_feat_acts = einops.rearrange(
                all_feat_acts,
                "batch seq feats -> feats (batch seq)",
            )
            flat_ignore_tokens_mask = einops.rearrange(
                ignore_tokens_mask,
                "batch seq -> (batch seq)",
            )
            valid_token_count = int(flat_ignore_tokens_mask.sum().item())

            if self.cfg.log_performance:
                log_perf_event(
                    "packaging_shape_summary",
                    batch=feature_batch_index,
                    feature_count=len(features),
                    token_shape=list(tokens.shape),
                    valid_token_count=valid_token_count,
                )

            with timed_stage(
                self.cfg.log_performance,
                "feature_statistics_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                feature_stats_input = flat_all_feat_acts[:, flat_ignore_tokens_mask]
                if feature_stats_input.shape[-1] == 0:
                    feature_stats_input = torch.zeros(
                        (flat_all_feat_acts.shape[0], 1),
                        dtype=flat_all_feat_acts.dtype,
                        device=flat_all_feat_acts.device,
                    )
                feature_stats = FeatureStatistics.create(
                    data=feature_stats_input,
                    batch_size=self.cfg.quantile_feature_batch_size,
                )

            feature_data_dict: dict[int, FeatureData] = {
                feat: FeatureData() for feat in features
            }
            layout = self.cfg.feature_centric_layout

            with timed_stage(
                self.cfg.log_performance,
                "feature_table_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                feature_tables_data = get_features_table_data(
                    feature_out_dir=feature_out_dir,
                    corrcoef_neurons=corrcoef_neurons,
                    corrcoef_encoder=corrcoef_encoder,
                    n_rows=layout.feature_tables_cfg.n_rows,  # type: ignore
                )
                for row_index, feat in enumerate(features):
                    feature_data_dict[feat].feature_tables_data = FeatureTablesData(
                        **{name: values[row_index] for name, values in feature_tables_data.items()}  # type: ignore
                    )

            if batch_dfa_results:
                for feature_idx, feature_data in batch_dfa_results.items():
                    all_consolidated_dfa_results[feature_idx].update(feature_data)

            with timed_stage(
                self.cfg.log_performance,
                "logits_histogram_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                for row_index, (feat, logit_vector) in enumerate(zip(features, logits)):
                    feature_data_dict[feat].logits_histogram_data = (
                        LogitsHistogramData.from_data(
                            data=logit_vector.to(torch.float32),
                            n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                            tickmode="5 ticks",
                            title=None,
                        )
                    )

            with timed_stage(
                self.cfg.log_performance,
                "activation_histogram_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                for row_index, feat in enumerate(features):
                    feat_acts = all_feat_acts[..., row_index]
                    masked_feat_acts = feat_acts * ignore_tokens_mask
                    nonzero_feat_acts = masked_feat_acts[masked_feat_acts > 0]
                    valid_feature_token_count = max(
                        1,
                        int(ignore_tokens_mask.sum().item()),
                    )
                    histogram_title = (
                        "ACTIVATIONS<br>DENSITY = "
                        f"{nonzero_feat_acts.numel() / valid_feature_token_count:.3%}"
                    )
                    if nonzero_feat_acts.numel() == 0:
                        feature_data_dict[feat].acts_histogram_data = ActsHistogramData(
                            title=histogram_title
                        )
                    else:
                        feature_data_dict[feat].acts_histogram_data = (
                            ActsHistogramData.from_data(
                                data=nonzero_feat_acts.to(torch.float32),
                                n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                                tickmode="5 ticks",
                                title=histogram_title,
                            )
                        )

            with timed_stage(
                self.cfg.log_performance,
                "logits_table_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                for feat, logit_vector in zip(features, logits):
                    feature_data_dict[feat].logits_table_data = get_logits_table_data(
                        logit_vector=logit_vector,
                        n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
                    )

            with timed_stage(
                self.cfg.log_performance,
                "sequence_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(features),
            ):
                for row_index, feat in enumerate(features):
                    feature_data_dict[feat].sequence_data = (
                        sequence_data_generator.get_sequences_data(
                            feat_acts=all_feat_acts[..., row_index] * ignore_tokens_mask,
                            feat_logits=logits[row_index],
                            resid_post=torch.tensor([]),
                            feature_resid_dir=feature_resid_dir[row_index],
                        )
                    )
                    if self.cfg.use_dfa:
                        feature_data_dict[feat].dfa_data = all_consolidated_dfa_results.get(
                            feat,
                            None,
                        )
                        feature_data_dict[feat].decoder_weights_data = (
                            get_decoder_weights_distribution(encoder, model, feat)[0]
                        )
                    if progress is not None:
                        progress[1].update(1)

            artifact_path = self._write_sequence_replay_artifact(
                feature_batch_index=feature_batch_index,
                features=features,
                tokens=tokens,
                selection_mask=ignore_tokens_mask,
                valid_token_count=valid_token_count,
                all_feat_acts=all_feat_acts,
                logits=logits,
                feature_resid_dir=feature_resid_dir,
            )
            if artifact_path is not None and self.cfg.log_performance:
                log_perf_event(
                    "sequence_replay_artifact",
                    batch=feature_batch_index,
                    path=artifact_path,
                    feature_count=len(features),
                    valid_token_count=valid_token_count,
                )
            new_feature_data = SaeVisData(
                cfg=self.cfg,
                feature_data_dict=feature_data_dict,
                feature_stats=feature_stats,
            )
            sae_vis_data.update(new_feature_data)

            if self.cfg.cleanup_each_minibatch:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if progress is not None:
            for pbar in progress:
                pbar.n = pbar.total

        if self.cfg.verbose:
            total_time = sum(time_logs.values())
            table = Table("Task", "Time", "Pct %")
            for task, duration in time_logs.items():
                table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
            rprint(table)

        sae_vis_data.cfg = self.cfg
        sae_vis_data.model = model
        sae_vis_data.encoder = encoder

        return sae_vis_data

    def set_seeds(self) -> None:
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        return None

    def _get_hf_unembed_matrix(self, model: AutoModelForCausalLM) -> Tensor:
        """Get the unembedding (lm_head) weight matrix from a HuggingFace model."""
        if hasattr(model, "lm_head"):
            return model.lm_head.weight.data.T  # (d_model, vocab_size)
        elif hasattr(model, "get_output_embeddings"):
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings.weight.data.T
        raise ValueError("Could not find unembedding matrix in HuggingFace model")

    def handle_features(
        self,
        features: Iterable[int] | None,
        encoder_wrapper: SAE,  # type: ignore
    ) -> list[int]:
        if features is None:
            return list(range(encoder_wrapper.cfg.d_sae))
        return list(features)

    def get_feature_batches(self, features_list: list[int]) -> list[list[int]]:
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
        return feature_batches

    def get_progress_bar(
        self,
        tokens: Int[Tensor, "batch seq"],
        feature_batches: list[list[int]],
        features_list: list[int],
    ):
        if self.cfg.prompt_minibatch_schedule:
            n_token_batches = len(self.cfg.prompt_minibatch_schedule)
        else:
            n_token_batches = (
                1
                if self.cfg.minibatch_size_tokens is None
                else math.ceil(len(tokens) / self.cfg.minibatch_size_tokens)
            )

        totals = (n_token_batches * len(feature_batches), len(features_list))

        if self.cfg.verbose:
            progress = [
                tqdm(total=totals[0], desc="Forward passes to cache data for vis"),
                tqdm(total=totals[1], desc="Extracting vis data from cached data"),
            ]
        else:
            progress = None

        return progress


def get_decoder_weights_distribution(
    encoder: SAE,  # type: ignore
    model: HookedSAETransformer,
    feature_idx: Union[int, List[int]],
) -> List[DecoderWeightsDistribution]:
    if not isinstance(feature_idx, list):
        feature_idx = [feature_idx]

    distribs = []
    for feature in feature_idx:
        att_blocks = einops.rearrange(
            encoder.W_dec[feature, :],
            "(n_head d_head) -> n_head d_head",
            n_head=model.cfg.n_heads,
        ).to("cpu")
        decoder_weights_distribution = (
            att_blocks.norm(dim=1) / att_blocks.norm(dim=1).sum()
        )
        distribs.append(
            DecoderWeightsDistribution(
                model.cfg.n_heads,
                [float(x) for x in decoder_weights_distribution],
            )
        )

    return distribs