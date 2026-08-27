"""Which hook the runner captures at, when the SAE declares two names for it.

``hook_name`` is a TransformerLens convention; ``hf_hook_name`` is the module path resolved from the
artifact's own config. For some releases they denote different tensors, so which one is authoritative
is a correctness question rather than a formatting one.
"""

from __future__ import annotations

import pytest

from sae_dashboard.neuronpedia.neuronpedia_runner import resolve_capture_hook_name

TL_NAME = "blocks.0.hook_mlp_in"
HF_NAME = "model.layers.0.pre_feedforward_layernorm.output"


class _Metadata:
    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


@pytest.fixture(params=["attribute", "mapping"], ids=["attr-metadata", "dict-metadata"])
def metadata_factory(request):
    """SAE metadata is reached both ways in this codebase; the resolver must handle both."""
    return _Metadata if request.param == "attribute" else (lambda **fields: dict(fields))


def test_transformer_lens_path_keeps_the_tl_name(metadata_factory):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    assert resolve_capture_hook_name(meta, use_huggingface=False) == TL_NAME


def test_huggingface_path_prefers_the_declared_module_path(metadata_factory):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    assert resolve_capture_hook_name(meta, use_huggingface=True) == HF_NAME


def test_falls_back_when_the_loader_publishes_no_module_path(metadata_factory):
    """Only two of SAELens' loaders populate the field, so the fallback is the common case."""
    meta = metadata_factory(hook_name="blocks.0.hook_resid_post")
    assert resolve_capture_hook_name(meta, use_huggingface=True) == "blocks.0.hook_resid_post"


@pytest.mark.parametrize("empty", ["", None], ids=["empty-string", "none"])
def test_an_unpopulated_field_is_not_a_module_path(metadata_factory, empty):
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=empty)
    assert resolve_capture_hook_name(meta, use_huggingface=True) == TL_NAME


def test_switching_capture_location_is_announced(metadata_factory, capsys):
    """Silently changing where activations come from is the failure mode this whole area has."""
    meta = metadata_factory(hook_name=TL_NAME, hf_hook_name=HF_NAME)
    resolve_capture_hook_name(meta, use_huggingface=True)
    out = capsys.readouterr().out
    assert HF_NAME in out and TL_NAME in out


def test_no_announcement_when_the_names_agree(metadata_factory, capsys):
    meta = metadata_factory(hook_name=HF_NAME, hf_hook_name=HF_NAME)
    resolve_capture_hook_name(meta, use_huggingface=True)
    assert capsys.readouterr().out == ""
