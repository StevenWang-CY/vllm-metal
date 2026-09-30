# DFlash checkpoint qualification

This is the model-qualification stage of [RFC #825](https://github.com/vllm-project/vllm-metal/issues/825).
It provides a native Qwen3 DFlash block forward and a numerical comparison tool.
It does **not** enable `method="dflash"` in vLLM serving.

The first reference checkpoint is
[z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16/tree/b74e3a329c4d963783143b1e970d95b002be72bd),
paired with Qwen3-4B. It has **five draft layers** and a 16-position block:
one anchor and 15 predictions. It does not fulfill the RFC's preferred
three-layer milestone. Quantized target conversions borrow their own actual
embedding and output projection; comparisons use that same target precision.

## Model contract

- Checkpoint target layer ID `i` selects Hugging Face `hidden_states[i + 1]`.
  Intermediate entries are decoder outputs before final normalization; the
  final entry is **after** the target's final norm. Use `DFlashTargetCapture`
  to adapt the shared bridge's pre-norm outputs to this contract. It preserves
  order and duplicates, and applies the target norm only for a final-layer tap.
  The reference checkpoint's `[1, 9, 17, 25, 33]` maps to bridge indices
  `[2, 10, 18, 26, 34]` and needs no final normalization.
- Each block attends to the complete committed context and every position
  within its own block. Proposal logits come from slots 1 onward.
- The caller supplies the target projections and full-prefix features.
  The model stores no borrowed target weights or persistent KV state.
- The initial loader accepts a local, unsharded z-lab Qwen3 checkpoint with
  uniform FP32, FP16, or BF16 weights, full attention, and default RoPE.
  Incompatible tensor names, shapes, precision, non-finite weights, and
  unsupported checkpoint semantics fail explicitly.
- Target geometry checks establish structural compatibility. Use the target
  named by the checkpoint's model card; equal geometry alone does not establish
  tokenizer identity or training compatibility. `load_dflash` requires the
  target's configuration as `target_config` and checks it before loading weights.
- `draft_logits` accepts valid target token IDs, normally supplied by the target
  sampler, and does not read token values back to the CPU. Validate external
  token IDs with `draft.validate_anchors(anchors)` at the input boundary, outside
  the compiled or repeated draft forward. Shape and dtype checks remain in the
  forward; mask IDs and anchors are represented as int64 without narrowing.

## Reproduce the numerical comparison

Download these snapshots with the Hugging Face CLI; it prints each local path:

```bash
hf download mlx-community/Qwen3-4B-4bit \
    --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
hf download z-lab/Qwen3-4B-DFlash-b16 \
    --revision b74e3a329c4d963783143b1e970d95b002be72bd
```

Save the official MIT-licensed
[model_mlx.py at 07ebd93](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
locally, then run from the repository's development environment:

```bash
python -m tools.dflash_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --reference /path/to/model_mlx.py \
    --output /path/to/new-results.json
```

The tool uses both capture implementations and compares eager and compiled draft
logits and greedy proposal IDs at batch sizes 1 and 2, context lengths 17, 33, and
65, and block sizes 2, 5, and 16. It records exact equality separately from the
numerical tolerance (`atol=rtol=1e-3`), rejects non-finite or incomplete comparisons,
and fails on any proposal mismatch. The report includes snapshot paths, native and
reference source hashes, and library versions. Use a new output file for each run.
If a checkpoint selects the final target layer, the tool normalizes the reference
hook's output to match the PyTorch/Hugging Face contract and records that adjustment
as `reference_final_norm_applied`. Independent tests compare captured features with
an actual Hugging Face Qwen3 forward, including the final layer and compiled replay.

This is forward parity, not generated-sequence parity or a performance benchmark.
The independent small-model tests also compare against explicit CPU attention
math and check that a causal block mask produces a different result.

## Serving milestones

The qualification forward recomputes context K/V with MLX attention. Serving
integration must bind committed draft KV to scheduler-owned `KVCacheStorage`,
use scheduler lookahead for temporary block writes, and express the block mask
through Metal paged attention. The first experimental serving stage must reject
prefix caching, as requested in the RFC. Lifecycle correctness, generated-token
parity, and batch-1/batch-N serving measurements remain separate gates before
claiming DSpark support or acceleration.
