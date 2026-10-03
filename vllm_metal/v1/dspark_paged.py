# SPDX-License-Identifier: Apache-2.0
"""DSpark proposals over scheduler-owned committed and lookahead draft KV."""

from collections.abc import Callable, Sequence

import mlx.core as mx

from vllm_metal.attention.caches.storage import KVCacheStorage
from vllm_metal.v1.block_draft_paged import BlockDraftPagedCache
from vllm_metal.v1.dspark import DSparkModel


class DSparkPagedCache(BlockDraftPagedCache):
    """Reuse target-feature KV while predicting from all K block positions.

    The caller supplies scheduler-owned pages and commits only verified target
    features. Temporary DSpark block KV must be overwritten even for accepted
    tokens. This binding does not register a serving method or allocate pages.
    """

    def __init__(
        self,
        model: DSparkModel,
        storage: KVCacheStorage,
        layer_names: tuple[str, ...],
        *,
        max_model_len: int,
    ) -> None:
        super().__init__(
            model.backbone, storage, layer_names, max_model_len=max_model_len
        )
        self.draft_model = model

    def compile_draft(
        self, *, num_draft_tokens: int
    ) -> Callable[
        [mx.array, Sequence[tuple[Sequence[int], int]]],
        tuple[mx.array, mx.array, mx.array | None],
    ]:
        """Return IDs, corrected logits and raw confidence using exactly K slots.

        Validate external anchor values with ``draft_model.validate_anchors``
        before the repeated forward. Metadata validation remains in the graph.
        """
        model = self.draft_model
        return self.compile_block(
            width=num_draft_tokens,
            embed=lambda anchors: model.block_embeddings(anchors, num_draft_tokens),
            finish=lambda hidden, anchors: model.greedy_proposal(
                model.backbone.norm(hidden), anchors
            ),
        )
