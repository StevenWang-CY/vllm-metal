# SPDX-License-Identifier: Apache-2.0
"""Checkpoint and block-forward contracts, checked against independent math."""

import json
from dataclasses import asdict, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3

from vllm_metal.patches.aux_hidden_states import AuxHiddenStateCapture
from vllm_metal.v1.dflash import DFlashConfig, DFlashModel, load_dflash


def _config():
    return DFlashConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        num_target_layers=4,
        max_position_embeddings=128,
        block_size=8,
        mask_token_id=63,
        target_layer_ids=(3, 0, 2),
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def _raw(config=None):
    raw = asdict(config or _config())
    raw.update(architectures=["DFlashDraftModel"], model_type="qwen3")
    raw["dflash_config"] = {
        name: raw.pop(name) for name in ("mask_token_id", "target_layer_ids")
    }
    return raw


def _torch_forward(model, embeddings, features, *, causal=False):
    """Use explicit QK/softmax/V, half-rotation RoPE, and unfused MLP on CPU."""
    weights = {
        name: torch.from_numpy(np.array(value.astype(mx.float32)))
        for name, value in tree_flatten(model.parameters())
    }
    cfg = model.config

    def linear(x, name):
        return functional.linear(x, weights[name + ".weight"])

    def norm(x, name):
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
            * weights[name + ".weight"]
        )

    def rope(x, offset):
        pos = torch.arange(offset, offset + x.shape[2], dtype=torch.float32)
        freq = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2) / cfg.head_dim)
        angles = (pos[:, None] * freq)[None, None]
        left, right = x.chunk(2, dim=-1)
        return torch.cat(
            [
                left * angles.cos() - right * angles.sin(),
                left * angles.sin() + right * angles.cos(),
            ],
            dim=-1,
        )

    h = torch.from_numpy(np.array(embeddings.astype(mx.float32)))
    context = torch.cat(
        [torch.from_numpy(np.array(f.astype(mx.float32))) for f in features], dim=-1
    )
    context = norm(linear(context, "fc"), "hidden_norm")
    batch, width, _ = h.shape
    length = context.shape[1]
    for i in range(cfg.num_hidden_layers):
        prefix = f"layers.{i}"
        a = prefix + ".self_attn"
        x = norm(h, prefix + ".input_layernorm")
        q = norm(
            linear(x, a + ".q_proj").reshape(
                batch, width, cfg.num_attention_heads, cfg.head_dim
            ),
            a + ".q_norm",
        ).transpose(1, 2)
        ctx_k = norm(
            linear(context, a + ".k_proj").reshape(
                batch, length, cfg.num_key_value_heads, cfg.head_dim
            ),
            a + ".k_norm",
        ).transpose(1, 2)
        blk_k = norm(
            linear(x, a + ".k_proj").reshape(
                batch, width, cfg.num_key_value_heads, cfg.head_dim
            ),
            a + ".k_norm",
        ).transpose(1, 2)
        k = torch.cat([rope(ctx_k, 0), rope(blk_k, length)], dim=2)
        v = (
            torch.cat([linear(context, a + ".v_proj"), linear(x, a + ".v_proj")], dim=1)
            .reshape(batch, length + width, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        scores = (rope(q, length) @ k.transpose(-1, -2)) * cfg.head_dim**-0.5
        if causal:
            allowed = (
                torch.arange(length + width)[None]
                <= (length + torch.arange(width))[:, None]
            )
            scores = scores.masked_fill(~allowed, -torch.inf)
        attention = (
            (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, width, -1)
        )
        h = h + linear(attention, a + ".o_proj")
        x = norm(h, prefix + ".post_attention_layernorm")
        h = h + linear(
            functional.silu(linear(x, prefix + ".mlp.gate_proj"))
            * linear(x, prefix + ".mlp.up_proj"),
            prefix + ".mlp.down_proj",
        )
    return norm(h, "norm").numpy()


@pytest.mark.parametrize("batch,length,width", [(1, 1, 2), (1, 17, 8), (3, 7, 4)])
@pytest.mark.parametrize(
    "dtype,tolerance", [(mx.float32, 2e-5), (mx.float16, 5e-3), (mx.bfloat16, 4e-2)]
)
def test_block_forward_matches_independent_reference(
    batch, length, width, dtype, tolerance
):
    model = DFlashModel(_config())
    model.set_dtype(dtype)
    embeddings = mx.random.normal((batch, width, 32)).astype(dtype)
    features = tuple(
        mx.random.normal((batch, length, 32)).astype(dtype) for _ in range(3)
    )
    actual = model(embeddings, features)
    expected = _torch_forward(model, embeddings, features)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, atol=tolerance, rtol=tolerance
    )
    # A causal implementation must fail this oracle, even if shapes look right.
    causal = _torch_forward(model, embeddings, features, causal=True)
    assert np.max(np.abs(causal - expected)) > 0.05
    # Dense batches remain isolated and do not retain state between calls.
    for row in range(batch):
        single = model(
            embeddings[row : row + 1], tuple(f[row : row + 1] for f in features)
        )
        np.testing.assert_allclose(
            np.array(single.astype(mx.float32)),
            np.array(actual[row : row + 1].astype(mx.float32)),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize("tied", [False, True])
def test_capture_indices_and_borrowed_quantized_projection(tied):
    config = replace(_config(), intermediate_size=64)
    args = qwen3.ModelArgs.from_dict(
        {
            **asdict(config),
            "model_type": "qwen3",
            "num_hidden_layers": 4,
            "tie_word_embeddings": tied,
        }
    )
    target = qwen3.Model(args)
    nn.quantize(target, group_size=32, bits=4)
    mx.eval(target.parameters())
    capture = AuxHiddenStateCapture(target, config.capture_layer_ids)
    tokens = mx.array([[1, 4, 2, 7, 9]])
    _, features = capture.run(target, tokens)
    assert config.capture_layer_ids == (4, 1, 3)
    hidden = target.model.embed_tokens(tokens)
    from mlx_lm.models.base import create_attention_mask

    mask = create_attention_mask(hidden)
    outputs = []
    for layer in target.model.layers:
        hidden = layer(hidden, mask)
        outputs.append(hidden)
    for feature, index in zip(features, config.target_layer_ids, strict=True):
        np.testing.assert_array_equal(np.array(feature), np.array(outputs[index]))
    model = DFlashModel(config)
    before = dict(tree_flatten(model.parameters()))
    project = target.model.embed_tokens.as_linear if tied else target.lm_head
    logits = model.draft_logits(
        mx.array([3]),
        features,
        num_draft_tokens=3,
        embed=target.model.embed_tokens,
        project=project,
    )
    full = model(target.model.embed_tokens(mx.array([[3, 63, 63, 63]])), features)
    expected = project(full[:, 1:])
    np.testing.assert_array_equal(np.array(logits), np.array(expected))
    assert logits.shape == (1, 3, 64)
    assert dict(tree_flatten(model.parameters())).keys() == before.keys()
    assert not any("embed_tokens" in name or "lm_head" in name for name in before)
    # Including slot zero is the DSpark anchor objective, not this checkpoint.
    assert not np.allclose(
        np.array(logits[:, 0]),
        np.array(project(full[:, 0])),
    )


def test_config_preserves_capture_order_and_validates_target():
    config = DFlashConfig.from_dict(_raw())
    assert config == _config()
    target = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "vocab_size": 64,
        "num_hidden_layers": 4,
    }
    config.validate_target(target)
    for key in target:
        with pytest.raises(ValueError, match="target"):
            config.validate_target({**target, key: -1})


def test_batched_one_token_drafts_use_contiguous_quantized_head_inputs():
    model = DFlashModel(_config())
    model.set_dtype(mx.bfloat16)
    embedding = nn.Embedding(64, 32)
    embedding.set_dtype(mx.bfloat16)
    embedding = nn.QuantizedEmbedding.from_embedding(embedding, group_size=32, bits=4)
    features = tuple(mx.random.normal((2, 7, 32)).astype(mx.bfloat16) for _ in range(3))
    anchors = mx.array([2, 3])
    full = model(embedding(mx.array([[2, 63], [3, 63]])), features)
    expected = embedding.as_linear(mx.contiguous(full[:, 1:]))
    actual = model.draft_logits(
        anchors,
        features,
        num_draft_tokens=1,
        embed=embedding,
        project=embedding.as_linear,
    )
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


@pytest.mark.parametrize(
    "anchors,width",
    [
        (mx.array([1]), 0),
        (mx.array([1]), 8),
        (mx.array([1]), True),
        (mx.array([-1]), 1),
        (mx.array([64]), 1),
        (mx.array([1.0]), 1),
        (mx.array([[1]]), 1),
        (mx.array([], dtype=mx.int32), 1),
    ],
)
def test_invalid_proposal_inputs_fail_before_embedding(anchors, width):
    def unexpected(_):
        pytest.fail("Invalid proposal reached the target projections")

    with pytest.raises(ValueError):
        DFlashModel(_config()).draft_logits(
            anchors, (), num_draft_tokens=width, embed=unexpected, project=unexpected
        )


def test_projection_vocabulary_mismatch_is_rejected():
    embedding = nn.Embedding(64, 32)
    with pytest.raises(ValueError, match="vocabulary"):
        DFlashModel(_config()).draft_logits(
            mx.array([1]),
            [mx.zeros((1, 2, 32))] * 3,
            num_draft_tokens=1,
            embed=embedding,
            project=lambda h: mx.zeros((*h.shape[:-1], 63)),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 0),
        ("num_key_value_heads", 3),
        ("head_dim", 7),
        ("block_size", 1),
        ("block_size", 129),
        ("mask_token_id", 64),
        ("mask_token_id", -1),
        ("mask_token_id", True),
        ("target_layer_ids", ()),
        ("target_layer_ids", (4,)),
        ("target_layer_ids", (True,)),
        ("rms_norm_eps", float("nan")),
        ("rope_theta", float("inf")),
        ("num_hidden_layers", True),
    ],
)
def test_invalid_geometry_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(_config(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["DFlash2DraftModel"]),
        ("model_type", "llama"),
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("attention_dropout", 0.1),
        ("layer_types", ["full_attention", "sliding_attention"]),
        ("layer_types", ["full_attention"]),
        ("use_sliding_window", True),
        ("rope_scaling", {"factor": 2}),
        ("rope_parameters", {"rope_type": "yarn"}),
        ("quantization", {"bits": 4}),
        ("is_causal", True),
        ("sample_from_anchor", True),
        ("output_multiplier", 2),
    ],
)
def test_unsupported_checkpoint_semantics_fail_early(field, value):
    with pytest.raises(ValueError):
        DFlashConfig.from_dict({**_raw(), field: value})


def _checkpoint(tmp_path):
    model = DFlashModel(_config())
    weights = dict(tree_flatten(model.parameters()))
    (tmp_path / "config.json").write_text(json.dumps(_raw()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return model, weights


def test_loading_roundtrip_preserves_checkpoint_and_forward(tmp_path):
    model, weights = _checkpoint(tmp_path)
    loaded = load_dflash(tmp_path)
    for name, tensor in tree_flatten(loaded.parameters()):
        np.testing.assert_array_equal(np.array(tensor), np.array(weights[name]))
    embeddings = mx.random.normal((1, 4, 32))
    features = [mx.random.normal((1, 7, 32)) for _ in range(3)]
    np.testing.assert_array_equal(
        np.array(model(embeddings, features)), np.array(loaded(embeddings, features))
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "mixed_dtype",
        "nonfinite",
        "second_file",
        "index",
    ],
)
def test_loading_rejects_incompatible_weights(tmp_path, corruption):
    _, weights = _checkpoint(tmp_path)
    if corruption == "missing":
        weights.pop("fc.weight")
    elif corruption == "extra":
        weights["lm_head.weight"] = mx.ones((64, 32))
    elif corruption == "shape":
        weights["fc.weight"] = weights["fc.weight"][:, :-1]
    elif corruption == "dtype":
        weights["fc.weight"] = weights["fc.weight"].astype(mx.int32)
    elif corruption == "mixed_dtype":
        weights["fc.weight"] = weights["fc.weight"].astype(mx.float16)
    elif corruption == "nonfinite":
        weights["fc.weight"] = mx.full(weights["fc.weight"].shape, float("nan"))
    elif corruption == "second_file":
        mx.save_safetensors(str(tmp_path / "extra.safetensors"), {"x": mx.zeros(1)})
    else:
        (tmp_path / "model.safetensors.index.json").write_text("{}")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_dflash(tmp_path)


@pytest.mark.parametrize("bad", ["count", "shape", "batch", "empty", "limit", "block"])
def test_incompatible_forward_inputs_fail(bad):
    model = DFlashModel(_config())
    embeddings = mx.zeros((1, 4, 32))
    features = [mx.zeros((1, 7, 32)) for _ in range(3)]
    if bad == "count":
        features.pop()
    elif bad == "shape":
        features[0] = mx.zeros((1, 8, 32))
    elif bad == "batch":
        embeddings = mx.zeros((2, 4, 32))
    elif bad == "empty":
        features = [mx.zeros((1, 0, 32)) for _ in range(3)]
    elif bad == "limit":
        features = [mx.zeros((1, 126, 32)) for _ in range(3)]
    else:
        embeddings = mx.zeros((1, 9, 32))
    with pytest.raises(ValueError):
        model(embeddings, features)
