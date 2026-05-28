from __future__ import annotations

import torch
from torch import Tensor

from sae_dashboard.feature_data_generator import FeatureDataGenerator
from sae_dashboard.neuronpedia.legacy_json_cpu import utils_fns as legacy_utils_fns


class LegacyJSONCPUFeatureDataGenerator(FeatureDataGenerator):
    def _transfer_feature_acts_for_output(
        self,
        feature_acts_for_output: Tensor,
    ) -> Tensor:
        return feature_acts_for_output

    def _uses_preserved_legacy_feature_act_concat(self) -> bool:
        return True

    def _uses_detached_legacy_json_cpu_compatibility(self) -> bool:
        return getattr(self.cfg, "legacy_json_cpu_compatibility", "current") == "detached_legacy"

    def _create_corrcoef_neurons(
        self,
        *,
        correlation_device: torch.device,
    ):
        if self._uses_detached_legacy_json_cpu_compatibility():
            return legacy_utils_fns.RollingCorrCoef(device=correlation_device)
        return super()._create_corrcoef_neurons(
            correlation_device=correlation_device,
        )

    def _create_corrcoef_encoder(
        self,
        *,
        feature_indices: list[int],
        correlation_device: torch.device,
    ):
        if self._uses_detached_legacy_json_cpu_compatibility():
            return legacy_utils_fns.RollingCorrCoef(
                indices=feature_indices,
                with_self=True,
                device=correlation_device,
            )
        return super()._create_corrcoef_encoder(
            feature_indices=feature_indices,
            correlation_device=correlation_device,
        )