import gc
import importlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Union, cast

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
from sae_dashboard.neuronpedia.legacy_json_cpu import (
    sae_vis_runner as legacy_json_cpu_sae_vis_runner,
)
from sae_dashboard.neuronpedia.legacy_json_cpu.runner import (
    is_preserved_legacy_json_cpu_path,
)
from sae_dashboard.neuronpedia.legacy_json_cpu.sequence_data_generator import (
    LegacyJSONCPUSequenceDataGenerator,
)
from sae_dashboard.perf_logging import log_perf_event, timed_stage
from sae_dashboard.sae_vis_data import (
    SaeVisColumnarBatch,
    SaeVisColumnarData,
    SaeVisConfig,
    SaeVisData,
)
from sae_dashboard.sequence_data_generator import (
    SequenceCoordinateTable,
    SequenceDataGenerator,
)
from sae_dashboard.transformer_lens_wrapper import (
    ActivationConfig,
    TransformerLensWrapper,
)
from sae_dashboard.utils_fns import (
    FeatureStatistics,
    HistogramData,
    build_activation_histogram_titles,
    build_activation_histogram_titles_from_densities,
)


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
        if self.cfg.columnar_artifact_dir is not None:
            self.cfg.columnar_artifact_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_columnar_modules() -> tuple[Any, Any, Any]:
        try:
            pyarrow = importlib.import_module("pyarrow")
            pyarrow_ipc = importlib.import_module("pyarrow.ipc")
            pyarrow_parquet = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise RuntimeError(
                "Columnar dashboard output requires pyarrow to be installed."
            ) from exc
        return pyarrow, pyarrow_ipc, pyarrow_parquet

    @property
    def _columnar_enabled(self) -> bool:
        return self.cfg.dashboard_output_format == "columnar"

    @property
    def _preserved_legacy_json_cpu_enabled(self) -> bool:
        return is_preserved_legacy_json_cpu_path(self.cfg)

    @property
    def _columnar_suffix(self) -> str:
        return "arrow" if self.cfg.columnar_artifact_format == "arrow" else "parquet"

    @staticmethod
    def _replace_index_column(
        table: Any,
        *,
        column_name: str,
        new_column_name: str | None = None,
        values: list[int],
        pyarrow: Any,
    ) -> Any:
        column_index = table.schema.get_field_index(column_name)
        if column_index == -1:
            raise ValueError(f"Column {column_name!r} is missing from columnar table.")
        resolved_column_name = new_column_name or column_name
        return table.set_column(
            column_index,
            pyarrow.field(resolved_column_name, pyarrow.int64()),
            pyarrow.array(values, type=pyarrow.int64()),
        )

    def _write_columnar_table(
        self,
        table: Any,
        path: Path,
        *,
        pyarrow: Any,
        pyarrow_ipc: Any,
        pyarrow_parquet: Any,
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.columnar_artifact_format == "arrow":
            with pyarrow.OSFile(str(path), "wb") as sink:
                with pyarrow_ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
        else:
            pyarrow_parquet.write_table(table, str(path))
        return int(table.num_rows)

    def _write_columnar_record_batches(
        self,
        batches: Iterable[Any],
        path: Path,
        *,
        schema: Any,
        pyarrow: Any,
        pyarrow_ipc: Any,
        pyarrow_parquet: Any,
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        if self.cfg.columnar_artifact_format == "arrow":
            with pyarrow.OSFile(str(path), "wb") as sink:
                with pyarrow_ipc.new_file(sink, schema) as writer:
                    for batch in batches:
                        writer.write_batch(batch)
                        row_count += int(batch.num_rows)
        else:
            with pyarrow_parquet.ParquetWriter(str(path), schema) as writer:
                for batch in batches:
                    writer.write_batch(batch)
                    row_count += int(batch.num_rows)
        return row_count

    @staticmethod
    def _decode_token_ids(
        model: HookedSAETransformer, token_ids: list[int]
    ) -> list[str]:
        tokens = model.tokenizer.convert_ids_to_tokens(token_ids)  # type: ignore[attr-defined]
        if isinstance(tokens, str):
            return [tokens]
        return [str(token) for token in tokens]

    def _feature_statistics_arrow_table(
        self,
        *,
        feature_indices: list[int],
        feature_stats_input: Tensor,
        feature_stats: FeatureStatistics | None,
        pyarrow: Any,
    ) -> Any:
        if self.cfg.feature_statistics_backend == "arrow":
            table = FeatureStatistics.create_arrow_table(
                data=feature_stats_input,
                include_quantiles=False,
            )
            return self._replace_index_column(
                table,
                column_name="feature_index",
                values=feature_indices,
                pyarrow=pyarrow,
            )

        if feature_stats is None:
            raise ValueError(
                "feature_stats is required for object feature statistics tables."
            )
        return pyarrow.table(
            {
                "feature_index": pyarrow.array(feature_indices, type=pyarrow.int64()),
                "max": pyarrow.array(feature_stats.max),
                "frac_nonzero": pyarrow.array(feature_stats.frac_nonzero),
                "positive_density": pyarrow.array(feature_stats.frac_nonzero),
            }
        )

    @staticmethod
    def _feature_tables_arrow_table(
        *,
        feature_indices: list[int],
        feature_tables_data: dict[str, list[Any]],
        pyarrow: Any,
    ) -> Any:
        columns: dict[str, Any] = {
            "feature_index": pyarrow.array(feature_indices, type=pyarrow.int64())
        }
        for name, values in feature_tables_data.items():
            columns[name] = pyarrow.array(values)
        return pyarrow.table(columns)

    @staticmethod
    def _histogram_arrow_table_from_objects(
        *,
        feature_indices: list[int],
        histogram_rows: list[HistogramData],
        pyarrow: Any,
    ) -> Any:
        return pyarrow.table(
            {
                "feature_index": pyarrow.array(feature_indices, type=pyarrow.int64()),
                "bar_heights": pyarrow.array(
                    [row.bar_heights for row in histogram_rows]
                ),
                "bar_values": pyarrow.array([row.bar_values for row in histogram_rows]),
                "tick_vals": pyarrow.array([row.tick_vals for row in histogram_rows]),
                "title": pyarrow.array([row.title for row in histogram_rows]),
            }
        )

    @staticmethod
    def _logits_tables_arrow_table(
        *,
        feature_indices: list[int],
        logits_table_rows: list[Any],
        pyarrow: Any,
    ) -> Any:
        return pyarrow.table(
            {
                "feature_index": pyarrow.array(feature_indices, type=pyarrow.int64()),
                "bottom_token_ids": pyarrow.array(
                    [row.bottom_token_ids for row in logits_table_rows]
                ),
                "bottom_logits": pyarrow.array(
                    [row.bottom_logits for row in logits_table_rows]
                ),
                "top_token_ids": pyarrow.array(
                    [row.top_token_ids for row in logits_table_rows]
                ),
                "top_logits": pyarrow.array(
                    [row.top_logits for row in logits_table_rows]
                ),
            }
        )

    def _write_columnar_batch(
        self,
        *,
        feature_batch_index: int,
        feature_indices: list[int],
        feature_stats_input: Tensor,
        feature_stats: FeatureStatistics | None,
        feature_statistics_table: Any | None,
        feature_tables_data: dict[str, list[Any]],
        logits_histogram_table: Any,
        activation_histogram_table: Any,
        logits_table_rows: list[Any],
        sequence_coordinate_tables: dict[int, SequenceCoordinateTable],
        model: HookedSAETransformer,
    ) -> SaeVisColumnarBatch:
        if self.cfg.columnar_artifact_dir is None:
            raise ValueError(
                "columnar_artifact_dir must be set when dashboard_output_format='columnar'."
            )

        pyarrow, pyarrow_ipc, pyarrow_parquet = self._load_columnar_modules()
        artifact_dir = (
            self.cfg.columnar_artifact_dir / f"feature_batch_{feature_batch_index}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        row_counts: dict[str, int] = {}
        tables: dict[str, str] = {}

        if feature_statistics_table is None:
            feature_statistics_table = self._feature_statistics_arrow_table(
                feature_indices=feature_indices,
                feature_stats_input=feature_stats_input,
                feature_stats=feature_stats,
                pyarrow=pyarrow,
            )
        feature_statistics_name = f"feature_statistics.{self._columnar_suffix}"
        row_counts["feature_statistics"] = self._write_columnar_table(
            feature_statistics_table,
            artifact_dir / feature_statistics_name,
            pyarrow=pyarrow,
            pyarrow_ipc=pyarrow_ipc,
            pyarrow_parquet=pyarrow_parquet,
        )
        tables["feature_statistics"] = feature_statistics_name

        feature_tables_table = self._feature_tables_arrow_table(
            feature_indices=feature_indices,
            feature_tables_data=feature_tables_data,
            pyarrow=pyarrow,
        )
        feature_tables_name = f"feature_tables.{self._columnar_suffix}"
        row_counts["feature_tables"] = self._write_columnar_table(
            feature_tables_table,
            artifact_dir / feature_tables_name,
            pyarrow=pyarrow,
            pyarrow_ipc=pyarrow_ipc,
            pyarrow_parquet=pyarrow_parquet,
        )
        tables["feature_tables"] = feature_tables_name

        logits_histograms_name = f"logits_histograms.{self._columnar_suffix}"
        row_counts["logits_histograms"] = self._write_columnar_table(
            logits_histogram_table,
            artifact_dir / logits_histograms_name,
            pyarrow=pyarrow,
            pyarrow_ipc=pyarrow_ipc,
            pyarrow_parquet=pyarrow_parquet,
        )
        tables["logits_histograms"] = logits_histograms_name

        activation_histograms_name = f"activation_histograms.{self._columnar_suffix}"
        row_counts["activation_histograms"] = self._write_columnar_table(
            activation_histogram_table,
            artifact_dir / activation_histograms_name,
            pyarrow=pyarrow,
            pyarrow_ipc=pyarrow_ipc,
            pyarrow_parquet=pyarrow_parquet,
        )
        tables["activation_histograms"] = activation_histograms_name

        logits_tables_table = self._logits_tables_arrow_table(
            feature_indices=feature_indices,
            logits_table_rows=logits_table_rows,
            pyarrow=pyarrow,
        )
        logits_tables_name = f"logits_tables.{self._columnar_suffix}"
        row_counts["logits_tables"] = self._write_columnar_table(
            logits_tables_table,
            artifact_dir / logits_tables_name,
            pyarrow=pyarrow,
            pyarrow_ipc=pyarrow_ipc,
            pyarrow_parquet=pyarrow_parquet,
        )
        tables["logits_tables"] = logits_tables_name

        def decode_token_ids(token_ids: list[int]) -> list[str]:
            return self._decode_token_ids(model, token_ids)

        pad_token_id = getattr(model.tokenizer, "pad_token_id", None)  # type: ignore[attr-defined]
        if self.cfg.columnar_emit_sequence_rows:
            sequence_rows_name = f"sequence_rows.{self._columnar_suffix}"
            with timed_stage(
                self.cfg.log_performance,
                f"sequence_row_{self._columnar_suffix}_stream_write",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(feature_indices),
            ):
                row_counts["sequence_rows"] = self._write_columnar_record_batches(
                    (
                        sequence_coordinate_tables[
                            feature_index
                        ].to_sequence_row_arrow_record_batch(feature_index)
                        for feature_index in feature_indices
                    ),
                    artifact_dir / sequence_rows_name,
                    schema=SequenceCoordinateTable.sequence_row_arrow_schema(),
                    pyarrow=pyarrow,
                    pyarrow_ipc=pyarrow_ipc,
                    pyarrow_parquet=pyarrow_parquet,
                )
            tables["sequence_rows"] = sequence_rows_name

        activation_row_batches: list[Any] = []
        activation_copy_row_batches: list[Any] = []
        if (
            self.cfg.columnar_emit_activation_rows
            or self.cfg.columnar_emit_activation_copy_rows
        ):
            with timed_stage(
                self.cfg.log_performance,
                "activation_row_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(feature_indices),
            ):
                for feature_index in feature_indices:
                    activation_row_batch = sequence_coordinate_tables[
                        feature_index
                    ].to_activation_row_arrow_record_batch(
                        feature_index,
                        decode_token_ids,
                        pad_token_id=pad_token_id,
                    )
                    if self.cfg.columnar_emit_activation_rows:
                        activation_row_batches.append(activation_row_batch)
                    if self.cfg.columnar_emit_activation_copy_rows:
                        activation_copy_row_batches.append(
                            SequenceCoordinateTable.activation_copy_row_arrow_record_batch_from_activation_row_record_batch(
                                activation_row_batch,
                                model_id=self.cfg.columnar_activation_copy_model_id
                                or "",
                                layer=self.cfg.columnar_activation_copy_layer or "",
                                creator_id=self.cfg.columnar_activation_copy_creator_id
                                or "",
                                created_at=self.cfg.columnar_activation_copy_created_at
                                or datetime.now(timezone.utc)
                                .replace(tzinfo=None)
                                .isoformat(),
                                activation_id_prefix=self.cfg.columnar_activation_copy_id_prefix,
                            )
                        )

        if activation_row_batches:
            activation_rows_table = pyarrow.Table.from_batches(
                activation_row_batches,
                schema=SequenceCoordinateTable.activation_row_arrow_schema(),
            )
            activation_rows_name = f"activation_rows.{self._columnar_suffix}"
            row_counts["activation_rows"] = self._write_columnar_table(
                activation_rows_table,
                artifact_dir / activation_rows_name,
                pyarrow=pyarrow,
                pyarrow_ipc=pyarrow_ipc,
                pyarrow_parquet=pyarrow_parquet,
            )
            tables["activation_rows"] = activation_rows_name

        if activation_copy_row_batches:
            activation_copy_rows_table = pyarrow.Table.from_batches(
                activation_copy_row_batches,
                schema=SequenceCoordinateTable.activation_copy_row_arrow_schema(),
            )
            activation_copy_rows_name = f"activation_copy_rows.{self._columnar_suffix}"
            with timed_stage(
                self.cfg.log_performance,
                "activation_copy_row_packaging",
                device=self.device,
                batch=feature_batch_index,
                feature_count=len(feature_indices),
            ):
                row_counts["activation_copy_rows"] = self._write_columnar_table(
                    activation_copy_rows_table,
                    artifact_dir / activation_copy_rows_name,
                    pyarrow=pyarrow,
                    pyarrow_ipc=pyarrow_ipc,
                    pyarrow_parquet=pyarrow_parquet,
                )
            tables["activation_copy_rows"] = activation_copy_rows_name

        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dashboard_output_format": "columnar",
                    "columnar_artifact_format": self.cfg.columnar_artifact_format,
                    "feature_batch_index": feature_batch_index,
                    "row_counts": row_counts,
                    "tables": tables,
                }
            ),
            encoding="utf-8",
        )
        return SaeVisColumnarBatch(
            feature_batch_index=feature_batch_index,
            feature_indices=list(feature_indices),
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            row_counts=row_counts,
        )

    def _write_columnar_root_manifest(self, batches: list[SaeVisColumnarBatch]) -> Path:
        if self.cfg.columnar_artifact_dir is None:
            raise ValueError(
                "columnar_artifact_dir must be set when dashboard_output_format='columnar'."
            )
        manifest_path = self.cfg.columnar_artifact_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "dashboard_output_format": "columnar",
                    "columnar_artifact_format": self.cfg.columnar_artifact_format,
                    "batches": [
                        {
                            "artifact_dir": batch.artifact_dir.name,
                            "feature_batch_index": batch.feature_batch_index,
                            "feature_indices": batch.feature_indices,
                        }
                        for batch in batches
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

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

    def _run_columnar_feature_batch(
        self,
        *,
        feature_batch_index: int,
        features: list[int],
        tokens: Int[Tensor, "batch seq"],
        model: HookedSAETransformer,
        encoder: SAE[Any],
        unembed_matrix: Tensor,
        feature_data_generator: FeatureDataGenerator,
        sequence_data_generator: SequenceDataGenerator,
        progress: Any,
        all_consolidated_dfa_results: dict[int, dict[Any, Any]],
    ) -> SaeVisColumnarBatch:
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
            feature_stats: FeatureStatistics | None = None
            feature_statistics_table = None
            if self.cfg.feature_statistics_backend == "arrow":
                pyarrow, _, _ = self._load_columnar_modules()
                feature_statistics_table = self._feature_statistics_arrow_table(
                    feature_indices=[int(feature) for feature in features],
                    feature_stats_input=feature_stats_input,
                    feature_stats=None,
                    pyarrow=pyarrow,
                )
            else:
                feature_stats = FeatureStatistics.create(
                    data=feature_stats_input,
                    batch_size=self.cfg.quantile_feature_batch_size,
                    use_sparse_quantiles=True,
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

        logits_histogram_table = None

        with timed_stage(
            self.cfg.log_performance,
            "logits_histogram_packaging",
            device=self.device,
            batch=feature_batch_index,
            feature_count=len(features),
        ):
            if self.cfg.logits_histogram_backend == "arrow":
                pyarrow, _, _ = self._load_columnar_modules()
                logits_histogram_table = HistogramData.from_data_batch_arrow_table(
                    data=logits.to(torch.float32),
                    n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                    tickmode="5 ticks",
                    title=None,
                    backend="torch",
                )
                logits_histogram_table = self._replace_index_column(
                    logits_histogram_table,
                    column_name="row_index",
                    new_column_name="feature_index",
                    values=[int(feature) for feature in features],
                    pyarrow=pyarrow,
                )
            else:
                for feat, logit_vector in zip(features, logits):
                    feature_data_dict[feat].logits_histogram_data = (
                        LogitsHistogramData.from_data(
                            data=logit_vector.to(torch.float32),
                            n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
                            tickmode="5 ticks",
                            title=None,
                        )
                    )

        activation_histogram_table = None

        with timed_stage(
            self.cfg.log_performance,
            "activation_histogram_packaging",
            device=self.device,
            batch=feature_batch_index,
            feature_count=len(features),
        ):
            if (
                feature_statistics_table is not None
                and "positive_density" in feature_statistics_table.schema.names
            ):
                activation_histogram_titles = (
                    build_activation_histogram_titles_from_densities(
                        feature_statistics_table.column("positive_density").to_pylist()
                    )
                )
            else:
                activation_histogram_titles = build_activation_histogram_titles(
                    flat_all_feat_acts,
                    valid_mask=flat_ignore_tokens_mask,
                )
            masked_flat_all_feat_acts = flat_all_feat_acts * flat_ignore_tokens_mask.to(
                device=flat_all_feat_acts.device,
                dtype=flat_all_feat_acts.dtype,
            )
            if self.cfg.activation_histogram_backend in {"torch", "polars"}:
                pyarrow, _, _ = self._load_columnar_modules()
                activation_histogram_table = HistogramData.from_data_batch_arrow_table(
                    data=masked_flat_all_feat_acts,
                    n_bins=layout.act_hist_cfg.n_bins,  # type: ignore
                    tickmode="5 ticks",
                    title=None,
                    positive_only=True,
                    titles=activation_histogram_titles,
                    backend=self.cfg.activation_histogram_backend,
                )
                activation_histogram_table = self._replace_index_column(
                    activation_histogram_table,
                    column_name="row_index",
                    new_column_name="feature_index",
                    values=[int(feature) for feature in features],
                    pyarrow=pyarrow,
                )

            if activation_histogram_table is None:
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

        logits_table_rows: list[Any] = []

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
                logits_table_rows.append(feature_data_dict[feat].logits_table_data)

        sequence_coordinate_tables: dict[int, SequenceCoordinateTable] = {}

        with timed_stage(
            self.cfg.log_performance,
            "sequence_packaging",
            device=self.device,
            batch=feature_batch_index,
            feature_count=len(features),
        ):
            for row_index, feat in enumerate(features):
                masked_feat_acts = all_feat_acts[..., row_index] * ignore_tokens_mask
                sequence_coordinate_tables[feat] = (
                    sequence_data_generator.get_sequence_coordinate_table(
                        feat_acts=masked_feat_acts,
                        feat_logits=logits[row_index],
                        resid_post=torch.tensor([]),
                        feature_resid_dir=feature_resid_dir[row_index],
                        selection_mask=ignore_tokens_mask,
                        selection_backend=self.cfg.sequence_selection_backend,
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

        pyarrow, _, _ = self._load_columnar_modules()
        if feature_statistics_table is None:
            feature_statistics_table = self._feature_statistics_arrow_table(
                feature_indices=[int(feature) for feature in features],
                feature_stats_input=feature_stats_input,
                feature_stats=feature_stats,
                pyarrow=pyarrow,
            )
        if logits_histogram_table is None:
            logits_histogram_table = self._histogram_arrow_table_from_objects(
                feature_indices=[int(feature) for feature in features],
                histogram_rows=[
                    feature_data_dict[int(feature)].logits_histogram_data
                    for feature in features
                ],
                pyarrow=pyarrow,
            )
        if activation_histogram_table is None:
            activation_histogram_table = self._histogram_arrow_table_from_objects(
                feature_indices=[int(feature) for feature in features],
                histogram_rows=[
                    feature_data_dict[int(feature)].acts_histogram_data
                    for feature in features
                ],
                pyarrow=pyarrow,
            )
        return self._write_columnar_batch(
            feature_batch_index=feature_batch_index,
            feature_indices=[int(feature) for feature in features],
            feature_stats_input=feature_stats_input,
            feature_stats=feature_stats,
            feature_statistics_table=feature_statistics_table,
            feature_tables_data=feature_tables_data,
            logits_histogram_table=logits_histogram_table,
            activation_histogram_table=activation_histogram_table,
            logits_table_rows=logits_table_rows,
            sequence_coordinate_tables=sequence_coordinate_tables,
            model=model,
        )

    def _run_object_feature_batch(
        self,
        *,
        feature_batch_index: int,
        features: list[int],
        tokens: Int[Tensor, "batch seq"],
        model: HookedSAETransformer,
        encoder: SAE[Any],
        unembed_matrix: Tensor,
        feature_data_generator: FeatureDataGenerator,
        sequence_data_generator: SequenceDataGenerator,
        progress: Any,
        all_consolidated_dfa_results: dict[int, dict[Any, Any]],
    ) -> SaeVisData:
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
                use_sparse_quantiles=True,
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
            for feat, logit_vector in zip(features, logits):
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
                masked_feat_acts = all_feat_acts[..., row_index] * ignore_tokens_mask
                feature_data_dict[feat].sequence_data = (
                    sequence_data_generator.get_sequences_data(
                        feat_acts=masked_feat_acts,
                        feat_logits=logits[row_index],
                        resid_post=torch.tensor([]),
                        feature_resid_dir=feature_resid_dir[row_index],
                        selection_mask=ignore_tokens_mask,
                        selection_backend=self.cfg.sequence_selection_backend,
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

        return SaeVisData(
            cfg=self.cfg,
            feature_data_dict=feature_data_dict,
            feature_stats=feature_stats,
        )

    @torch.inference_mode()
    def run(
        self,
        encoder: SAE,  # type: ignore
        model: Union[HookedSAETransformer, AutoModelForCausalLM],
        tokens: Int[Tensor, "batch seq"],
        tokenizer: AutoTokenizer = None,  # Required for HuggingFace models
    ) -> SaeVisData | SaeVisColumnarData:
        self.set_seeds()

        encoder_cfg = getattr(encoder, "cfg", None)
        encoder_architecture = getattr(encoder_cfg, "architecture", None)
        if callable(encoder_architecture):
            encoder_architecture = encoder_architecture()

        if "CLTLayerWrapper" in str(type(encoder)) or encoder_architecture in ["temporal"]:
            print("SaeVisRunner: Skipping fold_W_dec_norm() for CLT wrapper.")
        else:
            encoder.fold_W_dec_norm()

        if "CLTLayerWrapper" in str(type(encoder)):
            print("SaeVisRunner: Skipping hook_z_reshaping_mode check for CLT wrapper.")
        elif encoder.hook_z_reshaping_mode:
            encoder.turn_off_forward_pass_hook_z_reshaping()

        sae_vis_data = SaeVisData(cfg=self.cfg)
        columnar_batches: list[SaeVisColumnarBatch] = []
        time_logs: defaultdict[str, float] = defaultdict(float)

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)
        progress = self.get_progress_bar(tokens, feature_batches, features_list)

        create_kwargs: dict[str, Any] = {}
        if self.cfg.use_huggingface:
            create_kwargs["tokenizer"] = tokenizer
        feature_data_generator = FeatureDataGeneratorFactory.create(
            self.cfg,
            model,
            encoder,
            tokens,
            **create_kwargs,
        )

        unembed_matrix = (
            self._get_hf_unembed_matrix(model)
            if self.cfg.use_huggingface
            else _resolve_unembed_matrix(model)
        )
        sequence_data_generator_cls = (
            LegacyJSONCPUSequenceDataGenerator
            if self._preserved_legacy_json_cpu_enabled
            else SequenceDataGenerator
        )
        sequence_data_generator = sequence_data_generator_cls(
            cfg=self.cfg,
            tokens=tokens,
            W_U=unembed_matrix,
        )

        all_consolidated_dfa_results: dict[int, dict[Any, Any]] = {
            feature_idx: {} for feature_idx in features_list
        }
        for feature_batch_index, features in enumerate(feature_batches):
            if self._columnar_enabled:
                columnar_batches.append(
                    self._run_columnar_feature_batch(
                        feature_batch_index=feature_batch_index,
                        features=features,
                        tokens=tokens,
                        model=model,
                        encoder=encoder,
                        unembed_matrix=unembed_matrix,
                        feature_data_generator=feature_data_generator,
                        sequence_data_generator=sequence_data_generator,
                        progress=progress,
                        all_consolidated_dfa_results=all_consolidated_dfa_results,
                    )
                )
                if self.cfg.cleanup_each_minibatch:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue

            if self._preserved_legacy_json_cpu_enabled:
                sae_vis_data.update(
                    legacy_json_cpu_sae_vis_runner.run_object_feature_batch(
                        self,
                        feature_batch_index=feature_batch_index,
                        features=features,
                        tokens=tokens,
                        model=cast(HookedSAETransformer, model),
                        encoder=encoder,
                        unembed_matrix=unembed_matrix,
                        feature_data_generator=feature_data_generator,
                        sequence_data_generator=cast(
                            LegacyJSONCPUSequenceDataGenerator,
                            sequence_data_generator,
                        ),
                        progress=progress,
                        all_consolidated_dfa_results=all_consolidated_dfa_results,
                    )
                )
            else:
                sae_vis_data.update(
                    self._run_object_feature_batch(
                        feature_batch_index=feature_batch_index,
                        features=features,
                        tokens=tokens,
                        model=cast(HookedSAETransformer, model),
                        encoder=encoder,
                        unembed_matrix=unembed_matrix,
                        feature_data_generator=feature_data_generator,
                        sequence_data_generator=cast(
                            SequenceDataGenerator,
                            sequence_data_generator,
                        ),
                        progress=progress,
                        all_consolidated_dfa_results=all_consolidated_dfa_results,
                    )
                )

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

        if self._columnar_enabled:
            manifest_path = self._write_columnar_root_manifest(columnar_batches)
            return SaeVisColumnarData(
                cfg=self.cfg,
                artifact_dir=self.cfg.columnar_artifact_dir or Path(),
                manifest_path=manifest_path,
                batches=columnar_batches,
            )

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
