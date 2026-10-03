# SPDX-License-Identifier: Apache-2.0
"""DFlash alignment and borrowed heads over shared paged block attention."""

from collections.abc import Callable, Sequence

import mlx.core as mx

from vllm_metal.v1.block_draft_paged import BlockDraftPagedCache


class DFlashPagedCache(BlockDraftPagedCache):
    def compile_draft(
        self,
        *,
        num_draft_tokens: int,
        embed: Callable[[mx.array], mx.array],
        project: Callable[[mx.array], mx.array],
    ) -> Callable[[mx.array, Sequence[tuple[Sequence[int], int]]], mx.array]:
        """Predict K tokens after the anchor; reserve K+1 temporary KV positions."""
        model = self.model
        model._validate_num_draft_tokens(num_draft_tokens)

        def finish(hidden, anchors):
            # Preserve the contiguous post-slice norm used by quantized heads.
            return model._project_logits(
                model.norm(hidden[:, 1:]), project, anchors.shape[0], num_draft_tokens
            )

        return self.compile_block(
            width=num_draft_tokens + 1,
            embed=lambda anchors: model._draft_embeddings(
                anchors, num_draft_tokens, embed
            ),
            finish=finish,
        )
