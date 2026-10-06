# SPDX-License-Identifier: Apache-2.0
"""Weight-only DSpark Q4 and independent unpacked-weight references."""

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from tests.test_dspark import config
from vllm_metal.v1.dspark import DSparkModel


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
