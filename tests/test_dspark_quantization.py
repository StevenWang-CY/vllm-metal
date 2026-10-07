# SPDX-License-Identifier: Apache-2.0
"""Weight-only DSpark Q4 and independent unpacked-weight references."""

import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from tests.test_dflash import _target_config
from tests.test_dspark import config, raw_config
from vllm_metal.v1 import dspark
from vllm_metal.v1.dspark import DSparkModel, load_dspark
from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.spec_decode import SpeculativeDecodeController


def dequantized_model(model):
    """Unpack affine nibbles independently of MLX's quantized matmul."""
    weights = dict(tree_flatten(model.parameters()))
    for path, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        packed = np.array(module.weight)
        nibbles = ((packed[..., None] >> (4 * np.arange(8))) & 15).reshape(
            packed.shape[0], -1
        )
        scales = np.repeat(np.array(module.scales.astype(mx.float32)), 64, axis=-1)
        biases = np.repeat(np.array(module.biases.astype(mx.float32)), 64, axis=-1)
        # Rounding unpacked weights to BF16 before multiplication would not
        # reproduce the quantized kernel's arithmetic.
        weights[f"{path}.weight"] = mx.array(
            nibbles * scales + biases, dtype=mx.float32
        )
        del weights[f"{path}.scales"], weights[f"{path}.biases"]
    reference = DSparkModel(model.config)
    reference.load_weights(list(weights.items()), strict=True)
    return reference


def make_model(dtype):
    cfg = config()
    cfg = replace(
        cfg,
        backbone=replace(
            cfg.backbone, hidden_size=64, intermediate_size=128, head_dim=64
        ),
    )
    model = DSparkModel(cfg)
    model.set_dtype(dtype)
    return model


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_q4_replaces_only_backbone_linears_and_vocabulary_projection(dtype):
    model = make_model(dtype)
    before = dict(tree_flatten(model.parameters()))
    selected = {
        path
        for path, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (path.startswith("backbone.layers.") or path == "lm_head")
    }
    model.quantize_draft_linears()
    quantized = {
        p: m for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
    }
    assert set(quantized) == selected
    assert len(quantized) == 7 * model.config.backbone.num_hidden_layers + 1
    converted_bytes = 0
    for module in quantized.values():
        assert (module.bits, module.group_size, module.mode) == (4, 64, "affine")
        assert module.weight.dtype == mx.uint32
        assert module.scales.dtype == module.biases.dtype == dtype
        converted_bytes += sum(v.nbytes for _, v in tree_flatten(module.parameters()))
    assert converted_bytes < sum(before[f"{p}.weight"].nbytes for p in selected) / 3
    for path, value in tree_flatten(model.parameters()):
        if path.rsplit(".", 1)[0] not in selected:
            # Retained embeddings/heads/norms/fusion stay the original arrays.
            assert value is before[path]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("draft_topk", [None, 8])
def test_q4_heads_and_compiled_replay_match_unpacked_weights(dtype, draft_topk):
    model = make_model(dtype)
    # Keep the candidate boundary separated across reduced-precision kernels.
    # General random projection math is checked independently below.
    model.lm_head.weight = mx.broadcast_to(
        (mx.arange(64).astype(dtype) / 1024)[:, None], (64, 64)
    )
    model.markov_head.markov_w1.weight = mx.eye(4, dtype=dtype)[mx.arange(64) % 4]
    correction = mx.zeros((64, 4), dtype=dtype)
    correction[mx.array([1, 2, 3, 0]), mx.arange(4)] = 8
    model.markov_head.markov_w2.weight = correction
    model.quantize_draft_linears()
    reference = dequantized_model(model)
    compiled = mx.compile(
        lambda h, a: model.greedy_proposal(h, a, draft_topk=draft_topk)
    )
    # Use distinct anchors and states on replay, including more than one request.
    for shift in (0, 7):
        hidden = mx.random.normal((2, 7, 64)).astype(dtype)
        hidden[:, :, 0] = 64
        anchors = mx.array([1 + shift, 3 + shift])
        actual = compiled(hidden, anchors)
        expected = reference.greedy_proposal(hidden, anchors, draft_topk=draft_topk)
        np.testing.assert_array_equal(np.array(actual[0]), np.array(expected[0]))
        for a, b in zip(actual[1:], expected[1:], strict=True):
            np.testing.assert_allclose(
                np.array(a.astype(mx.float32)),
                np.array(b.astype(mx.float32)),
                atol=0.02 if dtype == mx.bfloat16 else 0.003,
                rtol=0.02,
            )
    with pytest.raises(ValueError, match="checkpoint precision"):
        model.greedy_proposal(hidden.astype(mx.float32), anchors)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_all_q4_projections_match_independent_unpacked_weights(dtype):
    model = make_model(dtype)
    model.quantize_draft_linears()
    reference = dict(dequantized_model(model).named_modules())
    for path, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            dense = reference[path]
            inputs = mx.random.normal((2, 7, dense.weight.shape[-1])).astype(dtype)
            np.testing.assert_allclose(
                np.array(module(inputs).astype(mx.float32)),
                np.array(dense(inputs).astype(mx.float32)),
                atol=0.02 if dtype == mx.bfloat16 else 0.003,
                rtol=0.02,
            )


def test_q4_invalid_geometry_rejected_before_any_replacement():
    model = make_model(mx.float16)
    # The late down projection must be validated before earlier leaves change.
    model.backbone.layers[-1].mlp.down_proj.weight = mx.zeros(
        (64, 65), dtype=mx.float16
    )
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="divisible by 64"):
        model.quantize_draft_linears()
    assert all(v is before[p] for p, v in tree_flatten(model.parameters()))
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())


def test_q4_requires_floating_checkpoint_and_cannot_be_applied_twice():
    model = make_model(mx.float32)
    with pytest.raises(ValueError, match="FP16/BF16"):
        model.quantize_draft_linears()
    model.set_dtype(mx.bfloat16)
    model.quantize_draft_linears()
    before = dict(tree_flatten(model.parameters()))
    with pytest.raises(ValueError, match="unquantized"):
        model.quantize_draft_linears()
    assert all(v is before[p] for p, v in tree_flatten(model.parameters()))


def _write_checkpoint(path, model, damage=None):
    weights = {
        name.removeprefix("backbone."): value
        for name, value in tree_flatten(model.parameters())
    }
    if damage is not None:
        damage(weights)
    (path / "config.json").write_text(json.dumps(raw_config(model.config)))
    mx.save_safetensors(str(path / "model.safetensors"), weights)


@pytest.mark.parametrize(
    "dimensions,error",
    [
        ({"hidden_size": 96}, "hidden_size=96"),
        ({"intermediate_size": 130}, "intermediate_size=130"),
        (
            {"num_attention_heads": 6, "head_dim": 16},
            r"num_attention_heads \* head_dim=96",
        ),
    ],
)
def test_q4_loader_rejects_dimensions_before_allocating_or_loading(
    tmp_path, monkeypatch, dimensions, error
):
    cfg = make_model(mx.float16).config
    cfg = replace(cfg, backbone=replace(cfg.backbone, **dimensions))
    model = DSparkModel(cfg)
    model.set_dtype(mx.float16)
    _write_checkpoint(tmp_path, model)
    # These are valid floating checkpoints; alignment is specific to Q4.
    assert isinstance(
        load_dspark(tmp_path, target_config=_target_config(cfg.backbone)).lm_head,
        nn.Linear,
    )

    def fail(*args, **kwargs):
        pytest.fail("Q4 dimensions must be checked before allocating or loading")

    monkeypatch.setattr(dspark, "DSparkModel", fail)
    monkeypatch.setattr(mx, "load", fail)
    with pytest.raises(ValueError, match=error):
        load_dspark(
            tmp_path,
            target_config=_target_config(cfg.backbone),
            draft_quantization="q4",
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("quantization", [None, "q4"])
def test_loader_preserves_runtime_conversion_and_unaligned_output_widths(
    tmp_path, dtype, quantization
):
    cfg = make_model(dtype).config
    # Only the input dimension of each selected linear needs group alignment.
    cfg = replace(
        cfg,
        backbone=replace(
            cfg.backbone, head_dim=16, intermediate_size=192, vocab_size=65
        ),
    )
    expected = DSparkModel(cfg)
    expected.set_dtype(dtype)
    _write_checkpoint(tmp_path, expected)
    if quantization is not None:
        expected.quantize_draft_linears()
    actual = load_dspark(
        tmp_path,
        target_config=_target_config(cfg.backbone),
        draft_quantization=quantization,
    )
    assert not actual.training
    assert isinstance(actual.lm_head, nn.QuantizedLinear) == (quantization == "q4")
    expected_weights = dict(tree_flatten(expected.parameters()))
    actual_weights = dict(tree_flatten(actual.parameters()))
    assert actual_weights.keys() == expected_weights.keys()
    for name, weight in actual_weights.items():
        reference = expected_weights[name]
        assert weight.dtype == reference.dtype
        assert mx.array_equal(weight, reference).item(), name


@pytest.mark.parametrize("mode", [False, True, 4, "q8", {"bits": 4}])
def test_loader_rejects_unknown_quantization_before_reading_checkpoint(tmp_path, mode):
    with pytest.raises(ValueError, match="dspark_draft_quantization must be 'q4'"):
        load_dspark(tmp_path, target_config={}, draft_quantization=mode)


@pytest.mark.parametrize(
    "damage,error",
    [
        (lambda w: w.pop("markov_head.markov_w1.weight"), "tensor names"),
        (lambda w: w.update({"lm_head.weight": mx.zeros((64, 65))}), "shape or dtype"),
        (
            lambda w: w.update(
                {"norm.weight": mx.full((64,), float("nan"), dtype=mx.float16)}
            ),
            "non-finite",
        ),
    ],
)
def test_q4_loader_still_validates_weights_before_conversion(
    tmp_path, monkeypatch, damage, error
):
    model = make_model(mx.float16)
    _write_checkpoint(tmp_path, model, damage)

    def fail(*args, **kwargs):
        pytest.fail("Checkpoint validation must finish before Q4 conversion")

    monkeypatch.setattr(DSparkModel, "quantize_draft_linears", fail)
    with pytest.raises(ValueError, match=error):
        load_dspark(
            tmp_path,
            target_config=_target_config(model.config.backbone),
            draft_quantization="q4",
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("quantization", [None, "q4"])
def test_builder_loads_checkpoint_with_requested_quantization(
    tmp_path, monkeypatch, dtype, quantization
):
    model = make_model(dtype)
    _write_checkpoint(tmp_path, model)
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=64),
            quantization=None,
            max_model_len=32,
        ),
        enable_adaptive_verification=False,
        draft_sample_method="greedy",
        rejection_sample_method="standard",
        dspark_draft_topk=None,
        quantization=None,
        kv_cache_dtype=None,
        num_speculative_tokens=7,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=spec,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            additional_config={"dspark_draft_quantization": quantization}
            if quantization is not None
            else {},
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                to_dict=lambda: _target_config(model.config.backbone)
            )
        ),
        kv_cache_dtype=dtype,
        _spec_decode_controller=SpeculativeDecodeController(),
    )
    monkeypatch.setattr(DSparkProposer, "_checkpoint_path", lambda runner: tmp_path)
    proposer = DSparkProposer.build(runner)
    assert proposer.draft_model.embed_tokens.weight.dtype == dtype
    assert proposer.max_model_len == 32
    assert isinstance(proposer.draft_model.lm_head, nn.QuantizedLinear) == (
        quantization == "q4"
    )
