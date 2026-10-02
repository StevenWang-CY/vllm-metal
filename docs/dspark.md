# DSpark checkpoint qualification

This is the DSpark model-forward stage of [RFC #825](https://github.com/vllm-project/vllm-metal/issues/825).
It implements the Qwen3 DSpark Markov and confidence heads on the shared
full-context DFlash backbone. Serving integration, scheduler-owned DSpark KV,
confidence-based planning, and sampled verification remain subsequent work.
This module does not enable `--speculative-config '{"method":"dspark", ...}'`.

`vllm_metal/v1/dspark.py` adapts the [MIT-licensed DeepSpec implementation](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/LICENSE).
It retains DeepSpec's copyright and full MIT permission notice, following the
existing DFlash module's approach to third-party attribution.

## Forward contract

The initial checkpoint is
[`deepseek-ai/dspark_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/tree/3457dff1417cb84927f6098a5fcb7cee85c934b7),
paired with Qwen3-4B. It has five draft layers and a trained width of seven.
The loader checks target geometry before loading weights; use the trained pair,
since matching dimensions alone do not establish training/tokenizer compatibility.

- DSpark owns its trained embedding and output projection. Both load from the
  draft checkpoint, including when the target is quantized.
- To predict K tokens, the input block contains one anchor and K−1 masks.
  **Slot 0 produces the first proposal**, so K=1 still runs the anchor through
  the backbone. The supported range is 1 through the trained block width.
- Features cover the prefix immediately before the anchor. Use
  `DFlashTargetCapture(target, draft.config.backbone)` for the shared HF
  `hidden_states[layer_id + 1]` convention, including final normalization when
  a final-layer tap is configured. Embedding-output taps are not supported.
- Every block query attends the full committed prefix and the complete block.
  `DSparkModel.block_hidden` returns normalized states for all K positions.
- `greedy_proposal` adds the vanilla low-rank Markov correction sequentially:
  the first position uses the anchor; later positions use the actual preceding
  proposal. It returns IDs, corrected logits, and optional raw confidence logits.
  Confidence uses that same predecessor and does not truncate the proposal.
- The loader accepts local, unsharded, uniform FP32/FP16/BF16 safetensors and
  preserves their precision. Tensor names/shapes and finite weights are checked.
  DFlash and DSpark share these checks in `draft_checkpoint.py`.
  Quantized, gated/RNN-head, GIDD, scaled/partial-RoPE, and non-Qwen3 checkpoints
  are rejected. Confidence heads may be absent or may use hidden states with
  or without Markov embeddings.
- Validate external anchor IDs with `draft.validate_anchors(anchors)` outside
  compilation. The repeated forward checks metadata without a CPU token readback.
  Input features and block states must match the draft's compute precision.

## Reproduce forward parity

Use a local checkout of the [official DeepSpec reference](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)
at `005e03b81cec38b7da6399833d609ee89a2587f2`. The tool imports that local code;
it does not fetch or execute remote code automatically. Download these snapshots:

```bash
hf download mlx-community/Qwen3-4B-4bit --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
hf download deepseek-ai/dspark_qwen3_4b_block7 --revision 3457dff1417cb84927f6098a5fcb7cee85c934b7
python -m tools.dspark_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --reference /path/to/DeepSpec \
    --context-lengths 1 15 16 17 65 257 1025 \
    --output /path/to/new-results.json
```

The default compares both drafters in FP32 using the checkpoint's stored weights
and native mlx-lm target features. It checks eager and compiled greedy proposals,
normalized block states, corrected logits, and confidence against official
PyTorch eager attention at batches 1/2 and widths 1/3/7. It requires exact proposal
IDs and `atol=rtol=1e-3` for floating outputs, rejects incomplete/non-finite
comparisons, and records versions, source hashes, paths, and case results.
Compilation is reused across inputs. A failed run cannot leave a stale passing
report at the requested output path.

The qualified FP32 matrix passes 42 cases, with 231 proposal IDs identical in
each execution mode. This is forward equivalence, not generated-sequence parity
or a performance claim. Loading the target and two draft implementations requires
substantial host/unified memory; FP32 uses more memory than the stored BF16 weights.

`--dtype bfloat16` is an additional diagnostic with looser tensor tolerances
(`atol=0.25`, `rtol=0.02`) and the same exact-token requirement. It currently
**fails** the qualified pair at batch 1, context 17, K=3: the reference's first
divergent choice ties at 18.25, while MLX produces 18.5 versus 18.25. That choice
changes subsequent Markov corrections. BF16 token equivalence, FP16 checkpoint
qualification, and serving losslessness therefore remain unestablished.

Small independent CPU-math tests cover block attention, Markov recurrence,
confidence predecessor alignment, optional heads, malformed checkpoints, and
compiled replay. Run them with `pytest tests/test_dspark.py`.
