# SPDX-License-Identifier: Apache-2.0
"""Experimental synchronous DFlash serving with scheduler-owned draft KV."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.dflash import DFlashModel, DFlashTargetCapture, load_dflash
from vllm_metal.v1.dflash_paged import DFlashPagedCache
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import PagedDecodeSegment, SpeculativeDecodeController

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner


class DFlashProposer:
    """Commit target features, then draft one bidirectional anchor/mask block.

    Accepted draft tokens still need their target features projected next step:
    temporary block KV is never equivalent to committed context KV. Prefix
    caching is deliberately unsupported until that lifecycle is qualified.
    """

    def __init__(
        self,
        model: DFlashModel,
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
        controller: SpeculativeDecodeController,
    ) -> None:
        model._validate_num_draft_tokens(num_draft_tokens)
        if model.fc.weight.dtype not in (mx.float16, mx.bfloat16):
            raise ValueError("DFlash serving requires an FP16 or BF16 checkpoint")
        self.model = model
        self.max_model_len = model.config.max_position_embeddings
        self.num_draft_tokens = num_draft_tokens
        self.embed, self.project = embed, project
        self.controller = controller
        self.layer_names = tuple(
            f"dflash_layers.{i}.self_attn"
            for i in range(model.config.num_hidden_layers)
        )
        self.cache: DFlashPagedCache | None = None
        self._drafts: dict[
            int, Callable[[mx.array, Sequence[tuple[Sequence[int], int]]], mx.array]
        ] = {}
        self._group_index = 0
        self._valid_ends: dict[str, int] = {}

    @classmethod
    def build(cls, runner: MetalModelRunner) -> DFlashProposer:
        config = runner.vllm_config
        spec = config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if config.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "DFlash on Metal requires --no-enable-prefix-caching"
            )
        if (
            runner.is_hybrid
            or runner.is_mla
            or runner._is_vlm
            or runner._is_pooling
            or config.lora_config is not None
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "DFlash on Metal requires a single-device Qwen3 text target without LoRA"
            )
        from vllm_metal.config import get_config

        if get_config().turboquant:
            raise NotImplementedError("DFlash on Metal does not support TurboQuant")
        if config.cache_config.block_size not in KERNEL_BLOCK_SIZES:
            raise NotImplementedError(
                "DFlash on Metal requires a native attention block size"
            )
        path = get_model_download_path(
            spec.draft_model_config.model, revision=spec.draft_model_config.revision
        )
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                path,
                revision=spec.draft_model_config.revision,
                allow_patterns=[
                    "config.json",
                    "*.safetensors",
                    "*.safetensors.index.json",
                ],
            )
        model = load_dflash(path, target_config=runner.model_config.hf_config.to_dict())
        if model.fc.weight.dtype != runner.kv_cache_dtype:
            raise NotImplementedError(
                "DFlash on Metal requires matching target and draft activation precision"
            )
        target = runner._forward_model
        # DFlash is qualified only for native Qwen3; preserve its tied/untied
        # and quantized head without registering borrowed target parameters.
        project = (
            target.model.embed_tokens.as_linear
            if target.args.tie_word_embeddings
            else target.lm_head
        )
        proposer = cls(
            model,
            num_draft_tokens=spec.num_speculative_tokens,
            embed=runner._target_input_embeddings,
            project=project,
            controller=runner._spec_decode_controller,
        )
        proposer.max_model_len = min(
            proposer.max_model_len, spec.draft_model_config.max_model_len
        )
        return proposer

    def target_capture(self, target: nn.Module) -> DFlashTargetCapture:
        return DFlashTargetCapture(target, self.model.config)

    def kv_specs(self, block_size: int) -> dict[str, KVCacheSpec]:
        cfg = self.model.config
        return {
            name: FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=cfg.num_key_value_heads,
                head_size=cfg.head_dim,
                dtype=MLX_TO_TORCH_DTYPE[self.model.fc.weight.dtype],
            )
            for name in self.layer_names
        }

    def bind_cache(
        self, storage: KVCacheStorage, *, group_index: int, max_model_len: int
    ) -> None:
        self.cache = DFlashPagedCache(
            self.model,
            storage,
            self.layer_names,
            max_model_len=min(max_model_len, self.max_model_len),
        )
        self._group_index = group_index
        self._drafts.clear()
        self._valid_ends.clear()

    def needs_target_hidden_states(
        self, decode_segments: Sequence[PagedDecodeSegment], *, has_final_prefill: bool
    ) -> bool:
        return False

    def release_requests(self, req_ids: set[str]) -> None:
        for req_id in req_ids:
            self._valid_ends.pop(req_id, None)

    def profile(self, features: Sequence[mx.array], max_num_seqs: int) -> None:
        """Include captured-feature projection and block drafting in profiling."""
        model = self.model
        context = model.hidden_norm(model.fc(mx.concatenate(features, axis=-1)))
        mx.eval(context)
        for layer in model.layers:
            mx.eval(layer.self_attn.k_proj(context), layer.self_attn.v_proj(context))
        # Prefix storage is accounted for by KV specs. A one-token prefix
        # profiles the fixed block's activations and the borrowed target head.
        cfg = model.config
        prefix = [
            mx.zeros((max_num_seqs, 1, cfg.hidden_size), dtype=context.dtype)
            for _ in cfg.target_layer_ids
        ]
        mx.eval(
            model.draft_logits(
                mx.zeros((max_num_seqs,), dtype=mx.int32),
                prefix,
                num_draft_tokens=self.num_draft_tokens,
                embed=self.embed,
                project=self.project,
            )
        )

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        cache = self.cache
        if cache is None:
            raise RuntimeError("DFlash must bind scheduler KV before proposing")
        width = ctx.num_speculative_tokens
        if not 0 <= width <= self.num_draft_tokens:
            raise ValueError("DFlash draft width exceeds the configured token budget")
        features = ctx.target_aux_hidden_states
        if not features or any(f.shape[0] != ctx.cu_seqlens[-1] for f in features):
            raise RuntimeError(
                "DFlash requires target features for every packed input row"
            )
        spans: list[tuple[Sequence[int], int, int]] = []
        slices: list[slice] = []
        updates: dict[str, int] = {}

        def ingest(
            req_id: str,
            groups: Sequence[Sequence[int]],
            start: int,
            count: int,
            row: int,
        ) -> None:
            end = min(start + count, cache.max_model_len)
            if start >= end:
                return
            if start != self._valid_ends.get(req_id, 0):
                raise RuntimeError(
                    f"DFlash committed context is discontinuous for {req_id!r}"
                )
            spans.append((groups[self._group_index], start, end - start))
            slices.append(slice(row, row + end - start))
            updates[req_id] = end

        for segment, tokens in zip(
            ctx.decode_segments, ctx.decode_token_ids, strict=True
        ):
            # The anchor plus accepted drafts have committed target features;
            # the last sampled token becomes the NEXT block's anchor.
            ingest(
                segment.req_id,
                segment.block_ids,
                segment.cache_start_pos,
                len(tokens),
                segment.start_row,
            )
        for i, prefill in enumerate(ctx.prefill_reqs):
            ingest(
                prefill.req_id,
                prefill.block_ids,
                prefill.start_pos,
                len(prefill.token_ids),
                ctx.cu_seqlens[ctx.num_decode_segments + i],
            )
        if spans:
            cache.write_context(
                [mx.concatenate([f[s] for s in slices]) for f in features], spans
            )
            self._valid_ends.update(updates)

        eligible = self.controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
        )
        req_ids, anchors, rows = [], [], []
        if width:
            for req_id, state in eligible:
                # This stage does not qualify scheduler-invalid grammar drafts.
                # Keep constrained requests on the target's grammar sampler.
                if state.sampling_params.structured_outputs is not None:
                    continue
                end = self._valid_ends.get(req_id, 0)
                if end + width + 1 > cache.max_model_len:
                    continue
                if end != len(state.token_ids) - 1:
                    raise RuntimeError(
                        "DFlash anchor does not follow committed target features"
                    )
                req_ids.append(req_id)
                anchors.append(state.token_ids[-1])
                rows.append((state.block_ids[self._group_index], end))
        if not rows:
            mx.eval(*cache.storage.buffers)
            return None
        # The scheduler selects the next width from its batch-size schedule.
        # Keep a compiled callable for each encountered nonzero width. K=0
        # still commits features above, so later drafting needs no replay.
        if width not in self._drafts:
            self._drafts[width] = cache.compile_draft(
                num_draft_tokens=width, embed=self.embed, project=self.project
            )
        logits = self._drafts[width](mx.array(anchors, dtype=mx.int32), rows)
        tokens = mx.argmax(logits, axis=-1)
        mx.eval(tokens, *cache.storage.buffers)
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=tokens.tolist())
