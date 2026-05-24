from __future__ import annotations

import torch
from eindex import (
    eindex,  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
)
from jaxtyping import Float, Int
from torch import Tensor

from sae_dashboard.components import SequenceMultiGroupData
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceDataGenerator,
    SequenceSelectionBackend,
)


class LegacyJSONCPUSequenceDataGenerator(SequenceDataGenerator):
    def _selection_feat_acts(
        self,
        feat_acts: Float[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq"]:
        target_device = torch.device(self.cfg.device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            return feat_acts
        return feat_acts.to(device=target_device)

    @torch.inference_mode()
    def get_sequences_data(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy_json_cpu",
    ) -> SequenceMultiGroupData:
        del feature_resid_dir, selection_mask, selection_backend
        indices_dict, indices_bold, n_bold = self.get_indices_dict(
            self.buffer,
            self._selection_feat_acts(feat_acts),
        )
        indices_buf = self.get_indices_buf(
            indices_bold=indices_bold,
            seq_length=self.seq_length,
            n_bold=n_bold,
            padded_buffer_width=self.padded_buffer_width,
        )
        token_ids = eindex(
            self.tokens,
            indices_buf[:, 1:],
            "[n_bold seq 0] [n_bold seq 1]",
        )
        (
            _,
            feat_acts_coloring,
            _,
        ) = self.index_objects_for_ablation_experiments(
            token_ids=token_ids,
            tokens=self.tokens,
            feat_acts=feat_acts,
            resid_post=resid_post,
            indices_bold=indices_bold,
            indices_buf=indices_buf,
        )
        return self.package_sequences_data(
            token_ids=token_ids,
            feat_acts_coloring=feat_acts_coloring,
            feat_logits=feat_logits,
            indices_dict=indices_dict,
            indices_bold=indices_bold,
        )

    @torch.inference_mode()
    def get_sequence_coordinate_table(
        self,
        feat_acts: Float[Tensor, "batch seq"],
        feat_logits: Float[Tensor, "d_vocab"],
        resid_post: Float[Tensor, "batch seq d_model"],
        feature_resid_dir: Float[Tensor, "d_model"],
        selection_mask: Int[Tensor, "batch seq"] | None = None,
        selection_backend: SequenceSelectionBackend = "legacy_json_cpu",
    ) -> SequenceCoordinateTable:
        del selection_backend
        return SequenceDataGenerator.get_sequence_coordinate_table(
            self,
            feat_acts=feat_acts,
            feat_logits=feat_logits,
            resid_post=resid_post,
            feature_resid_dir=feature_resid_dir,
            selection_mask=selection_mask,
            selection_backend="legacy_json_cpu",
        )