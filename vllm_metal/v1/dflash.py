# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Z Lab
# Copyright (c) 2026 vLLM Metal contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Native Qwen3 DFlash block forward for checkpoint qualification.

Adapted from z-lab/dflash's model_mlx.py at 07ebd93db9f472af339b644bb70221ad8428328a.
This stateless forward recomputes context K/V; it does not allocate a serving
cache or register a vLLM speculative method. Paged cache integration is separate.
The caller owns the target embedding, output projection, and captured features.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import MLP
from safetensors import safe_open


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_target_layers: int
    max_position_embeddings: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    rms_norm_eps: float
    rope_theta: float

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {
                "target_layer_ids",
                "mask_token_id",
                "rms_norm_eps",
                "rope_theta",
            }:
                continue
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DFlash {field.name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("DFlash requires divisible GQA heads and an even head_dim")
        if self.block_size < 2 or self.block_size > self.max_position_embeddings:
            raise ValueError(
                "DFlash block_size must fit the context and include an anchor"
            )
        if (
            type(self.mask_token_id) is not int
            or not 0 <= self.mask_token_id < self.vocab_size
        ):
            raise ValueError("DFlash mask_token_id must be in the target vocabulary")
        if not self.target_layer_ids or any(
            type(i) is not int or not 0 <= i < self.num_target_layers
            for i in self.target_layer_ids
        ):
            raise ValueError("DFlash target_layer_ids must name target decoder outputs")
        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"DFlash {name} must be finite and positive")

    @property
    def capture_layer_ids(self) -> tuple[int, ...]:
        """Translate zero-based decoder outputs to the shared capture convention."""
        return tuple(i + 1 for i in self.target_layer_ids)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DFlashConfig:
        if (
            raw.get("architectures") != ["DFlashDraftModel"]
            or raw.get("model_type") != "qwen3"
        ):
            raise ValueError("Expected a z-lab Qwen3 DFlashDraftModel checkpoint")
        supported = {
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_sliding_window": False,
            "sliding_window": None,
            "rope_scaling": None,
            "rope_parameters": None,
            "quantization": None,
            "quantization_config": None,
            "is_causal": False,
            "sample_from_anchor": False,
        }
        for name, expected in supported.items():
            if raw.get(name, expected) != expected:
                raise ValueError(f"Unsupported DFlash {name}: {raw[name]!r}")
        draft = raw.get("dflash_config")
        if not isinstance(draft, dict):
            raise ValueError("DFlash checkpoint requires dflash_config")
        for name, expected in {
            "input_embedding_scale": 1.0,
            "output_multiplier": 1.0,
            "final_logit_softcapping": None,
            "sample_from_anchor": False,
        }.items():
            if (
                raw.get(name, expected) != expected
                or draft.get(name, expected) != expected
            ):
                raise ValueError(f"Unsupported DFlash {name}")
        try:
            values = {
                f.name: raw[f.name]
                for f in fields(cls)
                if f.name not in {"target_layer_ids", "mask_token_id", "block_size"}
            }
            values.update(
                target_layer_ids=tuple(draft["target_layer_ids"]),
                mask_token_id=draft["mask_token_id"],
                block_size=draft.get("block_size", raw.get("block_size")),
            )
            config = cls(**values)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Incomplete DFlash checkpoint configuration: {exc}"
            ) from exc
        if (
            "block_size" in draft
            and "block_size" in raw
            and draft["block_size"] != raw["block_size"]
        ):
            raise ValueError("Conflicting DFlash block_size values")
        if (
            raw.get("layer_types", ["full_attention"] * config.num_hidden_layers)
            != ["full_attention"] * config.num_hidden_layers
        ):
            raise ValueError("Only full-attention DFlash layers are supported")
        return config

    def validate_target(self, target: Mapping[str, Any]) -> None:
        """Check structural compatibility; the checkpoint still determines pairing."""
        expected = {
            "model_type": "qwen3",
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_target_layers,
        }
        for name, value in expected.items():
            if target.get(name) != value:
                raise ValueError(f"DFlash target {name} must be {value!r}")


class _Attention(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, context: mx.array, rope: nn.RoPE) -> mx.array:
        batch, width, _ = x.shape
        length = context.shape[1]
        q = self.q_norm(
            self.q_proj(x).reshape(batch, width, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        ck = self.k_norm(
            self.k_proj(context).reshape(batch, length, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        cv = (
            self.v_proj(context)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        bk = self.k_norm(
            self.k_proj(x).reshape(batch, width, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        bv = (
            self.v_proj(x)
            .reshape(batch, width, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        q, ck, bk = rope(q, offset=length), rope(ck), rope(bk, offset=length)
        # Every block query sees the entire committed context AND draft block.
        # This is bidirectional block attention, not the target's causal mask.
        output = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate([ck, bk], axis=2),
            mx.concatenate([cv, bv], axis=2),
            scale=self.scale,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, width, -1))


class _DecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, context: mx.array, rope: nn.RoPE) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), context, rope)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashModel(nn.Module):
    """Full-context block forward, with no persistent KV or borrowed parameters."""

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [_DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        embeddings: mx.array,
        features: Sequence[mx.array],
        *,
        logits_start: int = 0,
    ) -> mx.array:
        """Return normalized block states from logits_start onward.

        Features cover the full prefix starting at zero. Each feature is
        [batch, context length, hidden size], in capture order.
        There is no padding: every row in a call has the same context/block size.
        """
        if (
            embeddings.ndim != 3
            or embeddings.shape[0] < 1
            or embeddings.shape[2] != self.config.hidden_size
        ):
            raise ValueError(
                "DFlash embeddings must have shape [batch, block, hidden_size]"
            )
        if not 1 <= embeddings.shape[1] <= self.config.block_size:
            raise ValueError("DFlash block exceeds the trained block_size")
        if type(logits_start) is not int or not 0 <= logits_start < embeddings.shape[1]:
            raise ValueError("DFlash logits_start must select a nonempty block suffix")
        if len(features) != len(self.config.target_layer_ids):
            raise ValueError(
                "DFlash requires one feature tensor per configured target layer"
            )
        shape = features[0].shape
        if (
            len(shape) != 3
            or shape[0] != embeddings.shape[0]
            or shape[1] < 1
            or shape[2] != self.config.hidden_size
            or any(f.shape != shape for f in features)
        ):
            raise ValueError(
                "DFlash features must share [batch, context length, hidden_size]"
            )
        if shape[1] + embeddings.shape[1] > self.config.max_position_embeddings:
            raise ValueError("DFlash context and block exceed max_position_embeddings")
        context = self.hidden_norm(self.fc(mx.concatenate(features, axis=-1)))
        h = embeddings
        for layer in self.layers:
            h = layer(h, context, self.rope)
        # Slice before normalization, as in the reference. Normalization produces
        # contiguous rows for the borrowed head; a strided BF16 slice can select
        # a different quantized projection kernel for batched one-token drafts.
        return self.norm(h[:, logits_start:])

    def draft_logits(
        self,
        anchors: mx.array,
        features: Sequence[mx.array],
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
    ) -> mx.array:
        """Borrow the actual target projections and return slots 1..K only."""
        if (
            type(num_draft_tokens) is not int
            or not 1 <= num_draft_tokens < self.config.block_size
        ):
            raise ValueError("DFlash requires 1 <= num_draft_tokens < block_size")
        if (
            anchors.ndim != 1
            or anchors.size < 1
            or not mx.issubdtype(anchors.dtype, mx.integer)
        ):
            raise ValueError("DFlash anchors must be a nonempty integer vector")
        # Compare Python integers so a narrow anchor dtype cannot truncate the
        # vocabulary bound. Reject wide out-of-range IDs before normalizing.
        min_anchor = cast(int, anchors.min().item())
        max_anchor = cast(int, anchors.max().item())
        if min_anchor < 0 or max_anchor >= self.config.vocab_size:
            raise ValueError("DFlash anchor token is outside the target vocabulary")
        anchors = anchors.astype(mx.int64)
        masks = mx.full(
            (anchors.shape[0], num_draft_tokens),
            self.config.mask_token_id,
            dtype=mx.int64,
        )
        inputs = mx.concatenate([anchors[:, None], masks], axis=1)
        hidden = self(embed(inputs), features, logits_start=1)
        logits = project(hidden)
        if logits.shape != (anchors.shape[0], num_draft_tokens, self.config.vocab_size):
            raise ValueError("DFlash target projection has an incompatible vocabulary")
        return logits


def load_dflash(path: str | Path) -> DFlashModel:
    """Load an unpacked, single-file z-lab checkpoint from a local snapshot."""
    path = Path(path)
    config = DFlashConfig.from_dict(json.loads((path / "config.json").read_text()))
    files = sorted(path.glob("*.safetensors"))
    if len(files) != 1 or (path / "model.safetensors.index.json").exists():
        raise ValueError("DFlash qualification requires one unsharded safetensors file")
    model = DFlashModel(config)
    parameters = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    expected = {name: tuple(t.shape) for name, t in parameters}
    # Check headers before evaluating any checkpoint or randomly initialized weight.
    with safe_open(files[0], framework="numpy") as stream:
        if set(stream.keys()) != expected.keys():
            raise ValueError("DFlash checkpoint tensor names do not match the model")
        dtypes = set()
        for name, shape in expected.items():
            tensor = stream.get_slice(name)
            dtypes.add(tensor.get_dtype())
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() not in {
                "F16",
                "BF16",
                "F32",
            }:
                raise ValueError(f"Invalid DFlash tensor shape or dtype: {name}")
        if len(dtypes) != 1:
            raise ValueError(
                "DFlash qualification requires uniform checkpoint precision"
            )
    weights = cast(dict[str, mx.array], mx.load(files[0]))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    parameters = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    if not all(bool(mx.all(mx.isfinite(t))) for _, t in parameters):
        raise ValueError("DFlash checkpoint contains non-finite weights")
    return model
