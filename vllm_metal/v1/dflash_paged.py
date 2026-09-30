# SPDX-License-Identifier: Apache-2.0
"""Compiled DFlash blocks over scheduler-owned committed and lookahead KV."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.attention.impls.sdpa import _build_block_tables
from vllm_metal.metal import get_ops
from vllm_metal.v1.dflash import DFlashModel
from vllm_metal.v1.proposer import validate_scheduler_blocks


class DFlashPagedCache:
    """Bind the draft's views of the same allocation used by the target.

    The scheduler owns every physical page, including the temporary block.
    Only target-feature projections are committed context. Draft block K/V
    must be overwritten after verification, even when its tokens were accepted.
    """

    def __init__(
        self,
        model: DFlashModel,
        storage: KVCacheStorage,
        layer_names: tuple[str, ...],
        *,
        max_model_len: int,
    ) -> None:
        self.model = model
        self.storage = storage
        cfg = model.config
        if len(layer_names) != cfg.num_hidden_layers or any(
            name not in storage.specs
            or storage.specs[name].num_kv_heads != cfg.num_key_value_heads
            or storage.specs[name].head_size != cfg.head_dim
            or storage.specs[name].head_size_v != cfg.head_dim
            for name in layer_names
        ):
            raise ValueError("DFlash cache layers must match the checkpoint geometry")
        self.cache = MetalPagedKVCache.from_upstream(storage, layer_names)
        self.block_size = self.cache.block_size
        self.max_model_len = min(max_model_len, model.config.max_position_embeddings)
        if self.cache.dtype not in (mx.float16, mx.bfloat16):
            raise ValueError("Paged DFlash requires FP16 or BF16 draft KV")
        if self.cache.head_dim not in (64, 96, 128, 256, 512):
            raise ValueError("Paged DFlash requires a tiled-attention head size")
        if self.cache.dtype != model.fc.weight.dtype:
            raise ValueError("DFlash KV must preserve the checkpoint precision")

    def _slots(self, blocks: Sequence[int], start: int, count: int) -> list[int]:
        validate_scheduler_blocks(
            "DFlash", blocks, self.block_size, total_positions=start + count
        )
        used = (start + count + self.block_size - 1) // self.block_size
        if any(block < 0 or block >= self.cache.num_blocks for block in blocks[:used]):
            raise ValueError("DFlash scheduler block is outside the shared allocation")
        return [
            blocks[p // self.block_size] * self.block_size + p % self.block_size
            for p in range(start, start + count)
        ]

    def write_context(
        self,
        features: Sequence[mx.array],
        spans: Sequence[tuple[Sequence[int], int, int]],
    ) -> None:
        """Write packed committed feature rows at their absolute positions.

        Each span is (scheduler block IDs, first position, row count). Features
        contain exactly those rows, in span order, without rejected target rows.
        """
        positions: list[int] = []
        slots: list[int] = []
        for blocks, start, count in spans:
            if start < 0 or count < 1 or start + count > self.max_model_len:
                raise ValueError("DFlash committed span exceeds the context limit")
            slots.extend(self._slots(blocks, start, count))
            positions.extend(range(start, start + count))
        if not positions:
            return
        cfg = self.model.config
        if len(features) != len(cfg.target_layer_ids) or any(
            f.shape != (len(positions), cfg.hidden_size) for f in features
        ):
            raise ValueError("DFlash features must match the committed span rows")
        context = self.model.hidden_norm(
            self.model.fc(mx.concatenate(features, axis=-1))
        )
        offsets = mx.array(positions, dtype=mx.int32)
        slot_mapping = mx.array(slots, dtype=mx.int64)
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            keys = attn.k_norm(
                attn.k_proj(context).reshape(-1, cfg.num_key_value_heads, cfg.head_dim)
            )
            # Treat packed tokens as a batch of one-token rows, allowing each
            # request's absolute position to be passed to native RoPE.
            keys = self.model.rope(keys[:, :, None, :], offset=offsets)[:, :, 0, :]
            values = attn.v_proj(context).reshape(keys.shape)
            keys, values = get_ops().reshape_and_cache(
                keys,
                values,
                self.cache.key_caches[i],
                self.cache.value_caches[i],
                slot_mapping,
            )
            self.cache.replace_layer_cache(i, keys, values)

    def compile_draft(
        self,
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
    ) -> Callable[[mx.array, Sequence[tuple[Sequence[int], int]]], mx.array]:
        """Compile once per draft width; positions and block tables stay inputs.

        Unlike the qualification forward, this never gathers or reprojects the
        full prefix. Fixed-width block tables avoid retracing as context grows.
        Native scatter outputs are returned explicitly and rebound to storage
        so later target/draft forwards retain their dependencies.
        """
        model = self.model
        cfg = model.config
        model._validate_num_draft_tokens(num_draft_tokens)
        width = num_draft_tokens + 1
        table_width = (self.max_model_len + self.block_size - 1) // self.block_size

        def forward(anchors, keys, values, tables, offsets, slots, ranges):
            h = model._draft_embeddings(anchors, num_draft_tokens, embed)
            batch = anchors.shape[0]
            boundaries = mx.arange(batch + 1, dtype=mx.int32) * width
            lengths = offsets + width
            updated_keys, updated_values = [], []
            for i, layer in enumerate(model.layers):
                attn = layer.self_attn
                x = layer.input_layernorm(h)
                q, k, v = attn.project_block(x, model.rope, offsets)
                q = q.transpose(0, 2, 1, 3)
                k = k.transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3).reshape(
                    -1, cfg.num_key_value_heads, cfg.head_dim
                )
                k, v = get_ops().reshape_and_cache(
                    k.reshape(v.shape),
                    v,
                    keys[i],
                    values[i],
                    slots,
                )
                out = mx.array(0)
                # The existing explicit block-range interface is also used by
                # image attention. Its bounds are inclusive. Every block row
                # sees the committed prefix and every row in its own block.
                get_ops().paged_attention_primitive(
                    q.reshape(-1, cfg.num_attention_heads, cfg.head_dim),
                    k,
                    v,
                    cfg.num_key_value_heads,
                    attn.scale,
                    0.0,
                    tables,
                    lengths,
                    boundaries,
                    self.block_size,
                    self.max_model_len,
                    -1,
                    out,
                    mm_prefix_ranges=ranges,
                )
                h = h + attn.o_proj(out.reshape(batch, width, -1))
                h = h + layer.mlp(layer.post_attention_layernorm(h))
                updated_keys.append(k)
                updated_values.append(v)
            logits = model._project_logits(
                model.norm(h[:, 1:]), project, batch, num_draft_tokens
            )
            return logits, tuple(updated_keys), tuple(updated_values)

        compiled = mx.compile(forward)

        def draft(anchors, rows):
            model._validate_anchor_metadata(anchors)
            if len(rows) != anchors.shape[0]:
                raise ValueError("DFlash needs one block table per anchor")
            tables, offsets, slots, ranges = [], [], [], []
            for blocks, length in rows:
                if length < 1 or length + width > self.max_model_len:
                    raise ValueError("DFlash block exceeds the context limit")
                slots.extend(self._slots(blocks, length, width))
                used = (length + width + self.block_size - 1) // self.block_size
                table = list(blocks[:used])
                tables.append(table + [0] * (table_width - len(table)))
                offsets.append(length)
                ranges.extend([(length, length + width - 1)] * width)
            block_tables, kernel_block_size = _build_block_tables(
                tables, self.block_size
            )
            if kernel_block_size != self.block_size:
                raise ValueError("DFlash currently requires a native cache block size")
            logits, keys, values = compiled(
                anchors,
                tuple(self.cache.key_caches),
                tuple(self.cache.value_caches),
                block_tables,
                mx.array(offsets, dtype=mx.int32),
                mx.array(slots, dtype=mx.int64),
                mx.array(ranges, dtype=mx.int32),
            )
            for i, (k, v) in enumerate(zip(keys, values, strict=True)):
                self.cache.replace_layer_cache(i, k, v)
            return logits

        return draft
