# SPDX-License-Identifier: Apache-2.0
"""Speculators checkpoint taps and block-anchored attention semantics."""

import copy
import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from tests.test_dflash import _target_config, _torch_forward
from tests.test_dspark import array, checkpoint, config, raw_config
from vllm_metal.v1.dspark import DSparkConfig, DSparkModel, load_dspark


def speculators_config():
    raw = raw_config(config())
    backbone = {
        key: raw.pop(key)
        for key in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
        )
    }
    backbone.update(
        layer_types=["sliding_attention", "full_attention"],
        use_sliding_window=True,
        sliding_window=8,
    )
    del raw["target_layer_ids"], raw["num_target_layers"]
    raw.update(
        architectures=["DSparkDraftModel"],
        speculators_model_type="dspark",
        speculators_config={"algorithm": "dspark"},
        transformer_layer_config=backbone,
        aux_hidden_state_layer_ids=[1, 2, 3],
        draft_vocab_size=64,
        target_hidden_size=None,
        sample_from_anchor=True,
        sliding_window_non_causal=False,
    )
    return raw


def test_speculators_preserves_tuple_indices_and_attention_policy():
    raw = speculators_config()
    original = copy.deepcopy(raw)
    cfg = DSparkConfig.from_dict(raw, target_config=_target_config())
    assert raw == original
    assert cfg.backbone.target_layer_ids == (0, 1, 2)
    assert cfg.backbone.capture_layer_ids == (1, 2, 3)
    assert cfg.context_windows == (8, None)
    assert cfg.causal_blocks
    raw["sliding_window_non_causal"] = True
    assert not DSparkConfig.from_dict(raw, target_config=_target_config()).causal_blocks
    with pytest.raises(ValueError, match="target configuration"):
        DSparkConfig.from_dict(raw)


@pytest.mark.parametrize("architecture", ["DSparkDraftModel", "DSparkSpeculator"])
def test_speculators_full_attention_configuration(architecture):
    raw = speculators_config()
    raw["architectures"] = [architecture]
    raw["transformer_layer_config"].update(
        layer_types=["full_attention", "full_attention"],
        use_sliding_window=False,
        sliding_window=None,
    )
    cfg = DSparkConfig.from_dict(raw, target_config=_target_config())
    assert cfg.backbone == replace(config().backbone, target_layer_ids=(0, 1, 2))
    assert cfg.context_windows == (None, None)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "architectures", ["UnknownDraftModel"]),
        (None, "speculators_config", {"algorithm": "dflash"}),
        (None, "sample_from_anchor", False),
        (None, "draft_vocab_size", 32),
        (None, "target_hidden_size", 16),
        (None, "aux_hidden_state_layer_ids", [0, 1, 2]),
        (None, "aux_hidden_state_layer_ids", [1, 2, 4]),
        (None, "aux_hidden_state_layer_ids", [1, 1, 2]),
        (None, "aux_hidden_state_layer_ids", [1, True, 3]),
        (None, "sliding_window_non_causal", "false"),
        (None, "input_embedding_scale", 2),
        (None, "tie_word_embeddings", True),
        (None, "markov_head_type", "unknown"),
        (None, "markov_rank", 0),
        (None, "enable_confidence_head", "false"),
        ("transformer_layer_config", "hidden_size", 16),
        ("transformer_layer_config", "model_type", "qwen3_5"),
        ("transformer_layer_config", "layer_types", ["sliding_attention"]),
        (
            "transformer_layer_config",
            "layer_types",
            ["full_attention", "linear_attention"],
        ),
        ("transformer_layer_config", "sliding_window", 0),
        ("transformer_layer_config", "sliding_window", True),
        ("transformer_layer_config", "use_sliding_window", False),
        ("transformer_layer_config", "attention_bias", True),
        ("transformer_layer_config", "tie_word_embeddings", True),
        ("transformer_layer_config", "rope_scaling", {"rope_type": "yarn"}),
        ("transformer_layer_config", "rope_parameters", {"rope_type": "yarn"}),
    ],
)
def test_incompatible_speculators_semantics_rejected(section, key, value):
    raw = speculators_config()
    (raw if section is None else raw[section])[key] = value
    with pytest.raises(ValueError):
        DSparkConfig.from_dict(raw, target_config=_target_config())


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_speculators_load_preserves_weights_and_tap_order(tmp_path, dtype):
    cfg, weights = checkpoint(tmp_path, dtype=dtype)
    (tmp_path / "config.json").write_text(json.dumps(speculators_config()))
    model = load_dspark(tmp_path, target_config=_target_config(cfg.backbone))
    from mlx.utils import tree_flatten

    for name, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(
            array(value), array(weights[name.removeprefix("backbone.")])
        )
        assert value.dtype == dtype
    assert model.backbone.context_windows == (8, None)
    assert model.backbone.config.capture_layer_ids == (1, 2, 3)


@pytest.mark.parametrize("windows", [(8, 8), (None, 8)])
@pytest.mark.parametrize("causal_blocks", [False, True])
@pytest.mark.parametrize("length", [1, 8, 9, 17])
def test_window_forward_matches_independent_attention(windows, causal_blocks, length):
    model = DSparkModel(
        replace(config(), context_windows=windows, causal_blocks=causal_blocks)
    )
    features = [mx.random.normal((2, length, 32)) for _ in range(3)]
    anchors = mx.array([3, 5])
    expected = _torch_forward(
        model.backbone,
        model.block_embeddings(anchors, 7),
        features,
        context_windows=windows,
        causal_blocks=causal_blocks,
    )
    for forward in (model.block_hidden, mx.compile(model.block_hidden)):
        hidden = forward(anchors, features, num_draft_tokens=7)
        np.testing.assert_allclose(array(hidden), expected, atol=2e-5, rtol=2e-5)


def test_window_boundary_and_future_block_are_excluded():
    model = DSparkModel(replace(config(), context_windows=(8, 8), causal_blocks=True))
    features = [mx.random.normal((1, 17, 32)) for _ in range(3)]
    embeddings = model.block_embeddings(mx.array([3]), 7)
    expected = model.backbone(embeddings, features)
    # Positions 0..8 are outside the window; the left edge is position 9.
    excluded = [mx.concatenate([f[:, :9] + 100, f[:, 9:]], axis=1) for f in features]
    np.testing.assert_array_equal(
        array(model.backbone(embeddings, excluded)), array(expected)
    )
    included = [
        mx.concatenate([f[:, :9], f[:, 9:10] + 100, f[:, 10:]], axis=1)
        for f in features
    ]
    assert not np.allclose(array(model.backbone(embeddings, included)), array(expected))
    # A later block input cannot influence earlier block outputs, including
    # through the second draft layer.
    changed = mx.concatenate([embeddings[:, :4], embeddings[:, 4:] + 100], axis=1)
    actual = model.backbone(changed, features)
    np.testing.assert_array_equal(array(actual[:, :4]), array(expected[:, :4]))
