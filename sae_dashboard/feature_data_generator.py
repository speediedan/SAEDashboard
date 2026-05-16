import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from torch import Tensor, nn
from tqdm.auto import tqdm

from sae_dashboard.dfa_calculator import DFACalculator
from sae_dashboard.huggingface_model_wrapper import (
    HuggingFaceModelWrapper,
    to_resid_direction_hf,
)
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.transformer_lens_wrapper import (
    TransformerLensWrapper,
    to_resid_direction,
)
from sae_dashboard.utils_fns import RollingCorrCoef

Arr = np.ndarray

# Type alias for model wrapper types
ModelWrapperType = Union[TransformerLensWrapper, HuggingFaceModelWrapper]


@dataclass(frozen=True)
class PromptTokenMinibatch:
    prompt_indices: tuple[int, ...]
    tokens: Int[Tensor, "batch seq"]
    seq_length: int
    primary_acts_batch_size: int | None = None
    cache_key: str | None = None


class FeatureDataGenerator:
    def __init__(
        self,
        cfg: SaeVisConfig,
        tokens: Int[Tensor, "batch seq"],
        model: ModelWrapperType,
        encoder: SAE[Any],
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.full_sequence_length = int(tokens.shape[1])
        self.token_minibatches = self.batch_tokens(tokens)
        if self.cfg.cache_dir is not None:
            self._prepare_activation_cache_dir(tokens)

        # DFA is only supported for TransformerLens models
        if cfg.use_dfa:
            if isinstance(model, HuggingFaceModelWrapper):
                raise NotImplementedError(
                    "DFA (Direct Feature Attribution) is not yet supported for HuggingFace models. "
                    "Please use TransformerLens (use_huggingface=False) for DFA."
                )
            assert (
                "hook_z" in encoder.cfg.hook_name
            ), f"DFAs are only supported for hook_z, but got {encoder.cfg.hook_name}"
            self.dfa_calculator = DFACalculator(model.model, encoder)  # type: ignore
        else:
            self.dfa_calculator = None

    @torch.inference_mode()
    def batch_tokens(self, tokens: Int[Tensor, "batch seq"]) -> list[PromptTokenMinibatch]:
        if self.cfg.prompt_minibatch_schedule:
            token_minibatches: list[PromptTokenMinibatch] = []
            for schedule_entry in self.cfg.prompt_minibatch_schedule:
                prompt_indices = tuple(int(index) for index in schedule_entry.get("prompt_indices", []))
                if not prompt_indices:
                    continue
                seq_length = int(schedule_entry.get("seq_length", tokens.shape[1]))
                if seq_length <= 0 or seq_length > tokens.shape[1]:
                    raise ValueError(
                        f"Prompt minibatch seq_length {seq_length} is out of range for "
                        f"tokens shape {tuple(tokens.shape)}"
                    )
                primary_acts_batch_size = schedule_entry.get(
                    "primary_acts_batch_size", self.cfg.primary_acts_batch_size
                )
                if primary_acts_batch_size is not None:
                    primary_acts_batch_size = int(primary_acts_batch_size)
                prompt_index_tensor = torch.tensor(prompt_indices, dtype=torch.long)
                trimmed_tokens = tokens.index_select(0, prompt_index_tensor)[:, :seq_length].contiguous()
                token_minibatches.append(
                    PromptTokenMinibatch(
                        prompt_indices=prompt_indices,
                        tokens=trimmed_tokens,
                        seq_length=seq_length,
                        primary_acts_batch_size=primary_acts_batch_size,
                        cache_key=self._build_prompt_cache_key(
                            prompt_indices=prompt_indices,
                            seq_length=seq_length,
                            primary_acts_batch_size=primary_acts_batch_size,
                        ),
                    )
                )
            return token_minibatches

        # Get tokens into minibatches, for the fwd pass
        token_minibatches = (
            (tokens,) if self.cfg.minibatch_size_tokens is None else tokens.split(self.cfg.minibatch_size_tokens)
        )
        token_minibatches = list(token_minibatches)

        prompt_minibatch_specs: list[PromptTokenMinibatch] = []
        prompt_offset = 0
        for token_minibatch in token_minibatches:
            prompt_count = int(token_minibatch.shape[0])
            prompt_minibatch_specs.append(
                PromptTokenMinibatch(
                    prompt_indices=tuple(range(prompt_offset, prompt_offset + prompt_count)),
                    tokens=token_minibatch,
                    seq_length=int(token_minibatch.shape[1]),
                    primary_acts_batch_size=self.cfg.primary_acts_batch_size,
                )
            )
            prompt_offset += prompt_count

        return prompt_minibatch_specs

    @staticmethod
    def _build_prompt_cache_key(
        *,
        prompt_indices: tuple[int, ...],
        seq_length: int,
        primary_acts_batch_size: int | None,
    ) -> str:
        digest = hashlib.sha1()
        digest.update(np.asarray(prompt_indices, dtype=np.int32).tobytes())
        digest.update(f"{seq_length}:{primary_acts_batch_size}".encode("utf-8"))
        return digest.hexdigest()[:16]

    @staticmethod
    def _activation_cache_manifest_path(cache_dir: Path) -> Path:
        return cache_dir / "activation_cache_layout.json"

    @staticmethod
    def _build_activation_cache_layout_key(tokens: Tensor) -> str:
        digest = hashlib.sha1()
        token_array = tokens.detach().to("cpu").contiguous().numpy()
        digest.update(np.asarray(token_array.shape, dtype=np.int32).tobytes())
        digest.update(str(token_array.dtype).encode("utf-8"))
        digest.update(token_array.tobytes())
        return digest.hexdigest()[:16]

    def _expected_activation_cache_manifest(self, tokens: Tensor) -> dict[str, Any]:
        return {
            "cache_version": 1,
            "layout_key": self._build_activation_cache_layout_key(tokens),
            "token_shape": list(tokens.shape),
            "token_dtype": str(tokens.dtype),
            "prompt_minibatch_count": len(self.token_minibatches),
        }

    def _prepare_activation_cache_dir(self, tokens: Tensor) -> None:
        cache_dir = self.cfg.cache_dir
        if cache_dir is None:
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._activation_cache_manifest_path(cache_dir)
        expected_manifest = self._expected_activation_cache_manifest(tokens)
        existing_manifest: dict[str, Any] | None = None

        if manifest_path.is_file():
            try:
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                existing_manifest = None

        if existing_manifest != expected_manifest:
            for stale_cache_path in cache_dir.glob("model_activations_*.pt"):
                stale_cache_path.unlink()
            manifest_path.write_text(
                json.dumps(expected_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    @staticmethod
    def _pad_sequence_tensor(sequence_tensor: Tensor, *, target_seq_len: int) -> Tensor:
        current_seq_len = int(sequence_tensor.shape[1])
        if current_seq_len == target_seq_len:
            return sequence_tensor
        if current_seq_len > target_seq_len:
            raise ValueError(
                f"Cannot pad tensor with seq length {current_seq_len} into shorter target {target_seq_len}"
            )
        padded_tensor = sequence_tensor.new_zeros(
            (sequence_tensor.shape[0], target_seq_len, *sequence_tensor.shape[2:])
        )
        padded_tensor[:, :current_seq_len].copy_(sequence_tensor)
        return padded_tensor

    @staticmethod
    def _scatter_feature_act_chunk(
        destination: Tensor,
        chunk: Tensor,
        *,
        prompt_indices: tuple[int, ...],
    ) -> None:
        index_tensor = torch.tensor(prompt_indices, dtype=torch.long, device=destination.device)
        destination.index_copy_(0, index_tensor, chunk)

    def _forward_model_acts(
        self,
        minibatch_tokens: torch.Tensor,
        *,
        primary_acts_batch_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        if primary_acts_batch_size is None:
            primary_acts_batch_size = self.cfg.primary_acts_batch_size
        if (
            primary_acts_batch_size is None
            or primary_acts_batch_size <= 0
            or minibatch_tokens.shape[0] <= primary_acts_batch_size
        ):
            activation_dict = self.model.forward(
                minibatch_tokens.to("cpu"),
                return_logits=False,  # type: ignore[arg-type]
            )
            return activation_dict

        activation_dict: Dict[str, torch.Tensor] | None = None
        offset = 0
        for token_chunk in minibatch_tokens.split(primary_acts_batch_size):
            chunk_activations = self.model.forward(token_chunk.to("cpu"), return_logits=False)  # type: ignore[arg-type]
            if activation_dict is None:
                activation_dict = {
                    key: value.new_empty((minibatch_tokens.shape[0], *value.shape[1:]))
                    for key, value in chunk_activations.items()
                }
            chunk_size = token_chunk.shape[0]
            for key, value in chunk_activations.items():
                activation_dict[key][offset : offset + chunk_size].copy_(value)
            offset += chunk_size
            del chunk_activations

        assert activation_dict is not None
        return activation_dict

    @torch.inference_mode()
    def get_feature_data(  # type: ignore
        self,
        feature_indices: list[int],
        progress: list[tqdm] | None = None,  # type: ignore
    ):  # type: ignore
        all_dfa_results = {feature_idx: {} for feature_idx in feature_indices}
        total_prompt_count = sum(int(minibatch.tokens.shape[0]) for minibatch in self.token_minibatches)

        # Create objects to store the data for computing rolling stats
        corrcoef_neurons = RollingCorrCoef()
        corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)

        # Get encoder & decoder directions
        feature_out_dir = self.encoder.W_dec[feature_indices]  # [feats d_autoencoder]

        # Use appropriate to_resid_direction function based on model wrapper type
        if isinstance(self.model, HuggingFaceModelWrapper):
            feature_resid_dir = to_resid_direction_hf(
                feature_out_dir, self.model
            )  # [feats d_model]
        else:
            feature_resid_dir = to_resid_direction(
                feature_out_dir, self.model  # type: ignore
            )  # [feats d_model]
        all_feat_acts_tensor: Tensor | None = None

        # ! Compute & concatenate together all feature activations & post-activation function values
        for i, minibatch in enumerate(self.token_minibatches):
            model_activation_dict = self.get_model_acts(i, minibatch)
            primary_acts = model_activation_dict[
                self.model.activation_config.primary_hook_point  # type: ignore
            ].to(self.encoder.device)
            all_features_acts = None

            # For TopK, compute all activations first, then select features
            if self.encoder.cfg.architecture() in ["topk", "batchtopk", "temporal"] or isinstance(
                self.encoder.activation_fn, TopK
            ):
                # Get all features' activations
                all_features_acts = self.encoder.encode(primary_acts)
                feature_acts = all_features_acts[:, :, feature_indices].to(DTYPES[self.cfg.dtype])
            else:
                with FeatureMaskingContext(self.encoder, feature_indices):
                    feature_acts = self.encoder.encode(primary_acts).to(DTYPES[self.cfg.dtype])

            # Optionally filter out token positions whose hidden-state norm is
            # an extreme outlier relative to the median norm in this minibatch.
            # Mirrors dictionary_learning's `remove_high_norm` handling for
            # models (e.g. Qwen) with random high-norm activation sinks.
            # Ref: https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/pytorch_buffer.py#L220
            if self.cfg.ignore_high_activation_norm_multiple is not None:
                norms_bs = primary_acts.norm(dim=-1)  # [batch, seq]
                median_norm = norms_bs.median()
                high_norm_mask = (
                    norms_bs
                    > median_norm * self.cfg.ignore_high_activation_norm_multiple
                )
                if high_norm_mask.any():
                    # Zero out feature activations at high-norm positions so
                    # they are excluded from max activating examples,
                    # histograms, and sequence-level stats.
                    feature_acts = feature_acts.masked_fill(
                        high_norm_mask.unsqueeze(-1).to(feature_acts.device), 0
                    )
                    # Also zero out model activations at these positions so
                    # they don't skew the rolling correlation statistics.
                    primary_acts = primary_acts.masked_fill(
                        high_norm_mask.unsqueeze(-1).to(primary_acts.device), 0
                    )

            self.update_rolling_coefficients(
                model_acts=primary_acts,
                feature_acts=feature_acts,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
            )

            feature_acts_for_output = self._pad_sequence_tensor(
                feature_acts,
                target_seq_len=self.full_sequence_length,
            )
            feature_acts_cpu = feature_acts_for_output.to(device="cpu", dtype=torch.bfloat16)

            if all_feat_acts_tensor is None:
                all_feat_acts_tensor = torch.empty(
                    (
                        total_prompt_count,
                        self.full_sequence_length,
                        feature_acts_cpu.shape[-1],
                    ),
                    dtype=feature_acts_cpu.dtype,
                    device=feature_acts_cpu.device,
                )
            self._scatter_feature_act_chunk(
                all_feat_acts_tensor,
                feature_acts_cpu,
                prompt_indices=minibatch.prompt_indices,
            )

            # Calculate DFA
            if self.cfg.use_dfa and self.dfa_calculator:
                max_value_indices = torch.argmax(feature_acts, dim=1)
                batch_dfa_results = self.dfa_calculator.calculate(
                    model_activation_dict,
                    self.model.hook_layer,  # type: ignore
                    feature_indices,
                    max_value_indices,
                )
                for feature_idx, feature_data in batch_dfa_results.items():
                    for prompt_idx in range(feature_data.shape[0]):
                        global_prompt_idx = minibatch.prompt_indices[prompt_idx]
                        all_dfa_results[feature_idx][global_prompt_idx] = {
                            "dfaValues": feature_data[prompt_idx]["dfa_values"].tolist(),
                            "dfaTargetIndex": int(feature_data[prompt_idx]["dfa_target_index"]),
                            "dfaMaxValue": float(feature_data[prompt_idx]["dfa_max_value"]),
                        }

            # Update the 1st progress bar; fwd passes and sequence data dominate these computations.
            if progress is not None:
                progress[0].update(1)

            del feature_acts_for_output
            del feature_acts_cpu
            del feature_acts
            del primary_acts
            del model_activation_dict
            if all_features_acts is not None:
                del all_features_acts

        if all_feat_acts_tensor is None:
            all_feat_acts_tensor = torch.empty(0)

        return (
            all_feat_acts_tensor,
            torch.tensor([]),  # all_resid_post, no longer used
            feature_resid_dir,
            feature_out_dir,
            corrcoef_neurons,
            corrcoef_encoder,
            all_dfa_results,
        )

    @torch.inference_mode()
    def get_model_acts(
        self,
        minibatch_index: int,
        minibatch: PromptTokenMinibatch,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        A function that gets the model activations for a given minibatch of tokens.
        Uses np.memmap for efficient caching.
        """
        minibatch_tokens = minibatch.tokens
        cache_path: Path | None = None
        if self.cfg.cache_dir is not None:
            cache_name = f"model_activations_{minibatch_index}"
            if minibatch.cache_key is not None:
                cache_name += f"_{minibatch.cache_key}"
            cache_path = self.cfg.cache_dir / f"{cache_name}.pt"
            if use_cache and cache_path.exists():
                # Removed duplicate assignment. mmap=True enables memory-mapped file I/O which allows
                # lazy loading of tensors without reading the entire file into memory upfront, making
                # it faster for large cached activation files. weights_only=False allows loading the
                # full pickled objects which can be faster than the restricted loader when dealing with
                # complex tensor dictionaries (though less secure for untrusted files).
                activation_dict = torch.load(
                    cache_path, map_location="cpu", weights_only=False, mmap=True
                )
            else:
                activation_dict = self._forward_model_acts(
                    minibatch_tokens,
                    primary_acts_batch_size=minibatch.primary_acts_batch_size,
                )
                save_tensor_dict_torch(activation_dict, cache_path)
        else:
            activation_dict = self._forward_model_acts(
                minibatch_tokens,
                primary_acts_batch_size=minibatch.primary_acts_batch_size,
            )

        if not self.model._activation_shapes_match_tokens(activation_dict, minibatch_tokens):
            activation_dict = self._forward_model_acts(
                minibatch_tokens,
                primary_acts_batch_size=minibatch.primary_acts_batch_size,
            )
            if cache_path is not None:
                save_tensor_dict_torch(activation_dict, cache_path)
        return activation_dict

    @torch.inference_mode()
    def update_rolling_coefficients(
        self,
        model_acts: Float[Tensor, "batch seq d_in"],
        feature_acts: Float[Tensor, "batch seq feats"],
        corrcoef_neurons: RollingCorrCoef | None,
        corrcoef_encoder: RollingCorrCoef | None,
    ) -> None:
        """

        Args:
            model_acts: Float[Tensor, "batch seq d_in"]
                The activations of the model, which the SAE was trained on.
            feature_idx: list[int]
                The features we're computing the activations for. This will be used to index the encoder's weights.
            corrcoef_neurons: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between feature activations & neurons.
            corrcoef_encoder: Optional[RollingCorrCoef]
                The object storing the minimal data necessary to compute corrcoef between pairwise feature activations.
        """
        # Update the CorrCoef object between feature activation & neurons
        feature_acts_by_feature = einops.rearrange(feature_acts, "batch seq feats -> feats (batch seq)")
        if corrcoef_neurons is not None:
            corrcoef_neurons.update(
                feature_acts_by_feature,
                einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)"),
            )

        # Update the CorrCoef object between pairwise feature activations
        if corrcoef_encoder is not None:
            corrcoef_encoder.update(
                feature_acts_by_feature,
                feature_acts_by_feature,
            )


def save_tensor_dict_torch(tensor_dict: Dict[str, torch.Tensor], filename: Path):
    torch.save(tensor_dict, filename)


def load_tensor_dict_torch(filename: Path, device: str) -> Dict[str, torch.Tensor]:
    return torch.load(filename, map_location=torch.device(device))  # Directly load to GPU


class FeatureMaskingContext:
    def __init__(self, sae: SAE[Any], feature_idxs: List[int]):
        self.sae = sae
        self.feature_idxs = feature_idxs
        self.original_weight = {}

    def __enter__(self):
        ## W_dec
        self.original_weight["W_dec"] = getattr(self.sae, "W_dec")
        # mask the weight
        masked_weight = self.sae.W_dec[self.feature_idxs]
        # set the weight
        setattr(self.sae, "W_dec", nn.Parameter(masked_weight))

        ## W_enc
        self.original_weight["W_enc"] = getattr(self.sae, "W_enc")
        # mask the weight
        masked_weight = self.sae.W_enc[:, self.feature_idxs]
        # set the weight
        setattr(self.sae, "W_enc", nn.Parameter(masked_weight))

        # Handle architecture as either attribute or method
        architecture = self.sae.cfg.architecture
        if callable(architecture):
            architecture = architecture()

        if architecture in [
            "standard",
            "standard_transcoder",
            "transcoder",
            "skip_transcoder",
        ]:
            ## b_enc
            self.original_weight["b_enc"] = getattr(self.sae, "b_enc")
            # mask the weight
            masked_weight = self.sae.b_enc[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "b_enc", nn.Parameter(masked_weight))

        elif architecture in [
            "jumprelu",
            "jumprelu_transcoder",
            "jumprelu_skip_transcoder",
        ]:
            ## b_enc
            self.original_weight["b_enc"] = getattr(self.sae, "b_enc")
            # mask the weight
            masked_weight = self.sae.b_enc[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "b_enc", nn.Parameter(masked_weight))

            ## threshold
            self.original_weight["threshold"] = getattr(self.sae, "threshold")
            # mask the weight
            masked_weight = self.sae.threshold[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "threshold", nn.Parameter(masked_weight))

        elif architecture in ["gated", "gated_transcoder"]:
            ## b_gate
            self.original_weight["b_gate"] = getattr(self.sae, "b_gate")
            # mask the weight
            masked_weight = self.sae.b_gate[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "b_gate", nn.Parameter(masked_weight))

            ## r_mag
            self.original_weight["r_mag"] = getattr(self.sae, "r_mag")
            # mask the weight
            masked_weight = self.sae.r_mag[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "r_mag", nn.Parameter(masked_weight))

            ## b_mag
            self.original_weight["b_mag"] = getattr(self.sae, "b_mag")
            # mask the weight
            masked_weight = self.sae.b_mag[self.feature_idxs]  # type: ignore
            # set the weight
            setattr(self.sae, "b_mag", nn.Parameter(masked_weight))
        else:
            raise (ValueError("Invalid architecture"))

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore
        # set everything back to normal
        for key, value in self.original_weight.items():
            setattr(self.sae, key, value)
