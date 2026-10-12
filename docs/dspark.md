# DSpark experimental serving

DSpark greedy serving is an experimental stage of [RFC #825](https://github.com/vllm-project/vllm-metal/issues/825).
It uses the shared DFlash target-capture and committed-feature lifecycle with
scheduler-owned draft KV. DSpark's own embeddings and Markov head propose tokens;
the target verifies every proposal. Prefix reuse is supported in synchronous
serving. Confidence-based planning, sampled verification, and asynchronous
scheduling remain subsequent work.

`vllm_metal/v1/dspark.py` adapts the [MIT-licensed DeepSpec implementation](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/LICENSE).
It retains DeepSpec's copyright and full MIT permission notice, following the
existing DFlash module's approach to third-party attribution.

## Serve the trained pair

```bash
vllm serve mlx-community/Qwen3-4B-4bit \
    --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --speculative-config '{
      "method": "dspark",
      "model": "deepseek-ai/dspark_qwen3_4b_block7",
      "revision": "3457dff1417cb84927f6098a5fcb7cee85c934b7",
      "num_speculative_tokens": 7
    }'
```

Send plain greedy requests (`temperature=0`). Requests with sampling, penalties,
grammar constraints, or sample logprobs use the target without drafting.
The first stage requires a single-device Qwen3 text target without LoRA or
TurboQuant, native cache blocks of 8/16/32, and matching target/draft activation
precision. Draft weights load before memory profiling and share the device
budget with target weights, activations, and scheduler-owned KV.
Pre-quantized draft checkpoints and an explicit draft-cache precision that differs
from the target activation precision are rejected. An optional runtime Q4
conversion is described below.

Widths 1 through the checkpoint's trained width are supported. The optional
`num_speculative_tokens_per_batch_size` schedule can use zero to pause drafting;
verified target features still commit during a pause. A K-token proposal writes
exactly K draft slots (anchor plus K-1 masks), including at the context boundary.
Drafting stops when the selected span would exceed the effective target/draft
context limit. Cancellation and preemption discard logical feature coverage;
resumed prefills adopt the scheduler's new common prefix and commit the suffix
before drafting resumes.

Prefix reuse follows the shared [DFlash cache lifecycle](dflash.md#cache-lifecycle-and-validation).
Both target and draft groups must have the prefix; an independently missing
group forces suffix recomputation. Fallback and zero-width requests still commit
draft features, so they can populate reusable prefixes. The scheduler owns
hashing, shared pages, eviction, and cache reset. Its existing speculative
block-drop policy stays in effect, and temporary draft slots are never committed
prefix data. Use `--no-enable-prefix-caching` for a cold-cache comparison.

KV offloading is rejected during configuration: the Metal offloader does not
support restoring the target and draft cache groups. Remove `--kv-offloading-size`
and any offloading connector from `--kv-transfer-config` when using DSpark.

`enable_adaptive_verification`, non-greedy `draft_sample_method`, and nonstandard
`rejection_sample_method` are rejected rather than ignored.
The Metal compatibility bridge exempts only `MetalWorker` from the GPU V1
runner's DSpark prohibition; all other upstream runner checks remain active.

### Speculators checkpoints

Self-contained Qwen3 DSpark checkpoints in the Speculators format are also
supported. For example, pair `Qwen/Qwen3-4B` with
[`RedHatAI/Qwen3-4B-speculator.dspark`](https://huggingface.co/RedHatAI/Qwen3-4B-speculator.dspark/tree/bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23):

```bash
vllm serve Qwen/Qwen3-4B \
    --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --max-model-len 4096 \
    --no-async-scheduling \
    --speculative-config '{
      "method": "dspark",
      "model": "RedHatAI/Qwen3-4B-speculator.dspark",
      "revision": "bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23",
      "num_speculative_tokens": 7
    }'
```

The loader preserves this checkpoint's hidden-state tuple indices and attention
policy: each draft query sees up to 2,048 prefix positions before the anchor and
the preceding positions in its own block, including itself. The prefix window
stays anchored at the block start. Full-attention layers remain bidirectional;
`sliding_window_non_causal=true` also permits future positions within sliding
layers' blocks. KV gathers cover the longest live window in the batch, rounded
to bounded reusable shapes, so short requests avoid reading the full trained
window. They reuse scheduler-owned storage without changing page ownership,
prefix reuse, or eviction.

This path requires a full vocabulary, vanilla Markov head, interior target
hidden-state taps, and a single floating-point safetensors file containing all
draft weights, including embeddings and the output head. Checkpoints that omit
verifier-owned weights or use other architectures remain unsupported. Use the
checkpoint's configured block size (seven here); a model-card table alone does
not establish a different supported width. Acceptance depends on the trained
draft, target precision, prompts, and thinking mode. Measure the intended pair
with the same inputs and serving settings before making speedup claims.

Production qualification remains incomplete. Fresh speculative and target-only
greedy outputs can differ, including across batch shapes and prefix-cache states.
BF16 comparisons with the independent Speculators reference also have failures;
passing proposal or top-K checks alone does not establish numerical agreement.
Keep this path experimental while these checks remain unresolved.

### Limit Markov candidates

Optionally add `"dspark_draft_topk": 64` to the speculative configuration.
For each draft position, DSpark selects that many candidates from its base
logits and computes the sequential Markov correction only for those tokens.
This reduces Markov projection work but can exclude the full-vocabulary winner,
changing proposals and acceptance. The target still verifies every proposal
against its full vocabulary. Confidence uses the actual preceding draft token.

The limit must be an integer from 1 through the draft vocabulary size. An
explicit setting overrides the checkpoint's `dspark_draft_topk`; with neither
set, the existing full-vocabulary path is unchanged. A limit equal to the
vocabulary size also uses that path. Smaller limits trade candidate coverage for
less projection work; measure acceptance and end-to-end latency together.
This option does not enable adaptive verification or change the proposal width.

### Quantize draft linear layers

Add `--additional-config '{"dspark_draft_quantization":"q4"}'` to the serve
command to convert the draft backbone's linear layers and its vocabulary
projection to MLX affine 4-bit weights with group size 64. Conversion runs once,
after checkpoint validation and before memory profiling or compilation. Omit the
option to retain the checkpoint weights. Only `q4` is supported, and the option
requires `method="dspark"`.

Embeddings, feature fusion, normalization, and Markov/confidence heads retain
checkpoint precision. Draft activations and KV also retain FP16/BF16 precision;
the target is unchanged. This reduces resident draft weight memory, but startup
still loads the original floating checkpoint before conversion. Linear input
dimensions must be divisible by 64; incompatible checkpoint dimensions are
rejected before allocating the draft model or loading its weights.

Q4 can change draft proposals and acceptance. The target still verifies each
proposal against its full vocabulary. Measure acceptance and serving latency
together; reduced draft memory does not establish a speedup or native greedy
equivalence. The existing reduced-precision qualification limits still apply.

Both `tools.check_parity` and
`tools.benchmark.dspark_serving_benchmark` accept
`--dspark-draft-quantization q4`, including with `--dspark-draft-topk 64`.
The option applies only to the DSpark arm; native, target-only and ordinary
draft-model controls retain their existing settings.

## Serving validation

The shared lifecycle tests exercise both DFlash and DSpark, in both target
verification layouts. They require exact output IDs against target-only serving,
actual drafting and rejection, chunked prefill, context/page boundaries, mixed
greedy/fallback batches, preemption/recomputation, cancellation and request-ID
reuse, stop/EOS handling, and scheduler-driven width changes through zero.

```bash
pytest -m slow tests/test_block_draft_serving_e2e.py tests/test_block_draft_schedule_e2e.py
pytest -m slow tests/test_block_draft_prefix_caching_e2e.py -k dspark
python -m tools.check_parity \
    --method dspark --num-draft-tokens 7 \
    --model /path/to/target/snapshot --draft /path/to/draft/snapshot \
    --batch-size 1 2 --max-tokens 32 --output-dir /path/to/new-serving-results
```

The prefix-reuse tests pin the documented 4B target/DSpark revisions and
use K=7 with top-K=64 in both verification layouts. They require exact output IDs
against cache-disabled serving, observed cache hits and verified drafts after
reuse, reduced prefill work, fallback-produced prefixes, shared prompts,
cancellation, and cache-pressure resume. Per-case IDs, counters, and elapsed times
are retained in pytest's temporary directory; timings are observations rather
than performance gates. A retained-page preemption case requires a nonzero hit
on resume. A deliberately incorrect proposal exercises rejection without relying
on the model making a natural mistake; its output must still match the original
greedy continuation.
The first verification window also checks the sampled anchor, absolute position,
and first-proposal IDs against cache-disabled serving, so target corrections cannot
hide a draft shift after prefill or resume. Separately trained drafters are not
required to match an identical target/draft-model control's acceptance rate.

The shared parity tool compares native mlx-lm, target-only serving, and DSpark
serving. It records actual verification counts and reports `EXACT`, `TOP_K_MATCH`,
and failures separately; top-k agreement is not exact sequence equivalence.
These are correctness checks, not throughput or latency measurements.
Pass `--dspark-draft-topk 64` to the parity tool to exercise candidate limiting.

For the pinned pair above on Apple M5 Max, with vLLM 0.30.0, MLX 0.32.1,
and mlx-lm 0.32.0, the shared 40-prompt corpus (32 output tokens, K=7) reports:

| Mode | Batch size | EXACT vs native | TOP_K_MATCH | FAIL |
| --- | --- | --- | --- | --- |
| Target only | 1 | 27 | 13 | 0 |
| Target only | 2 | 27 | 13 | 0 |
| DSpark | 1 | 24 | 16 | 0 |
| DSpark | 2 | 28 | 12 | 0 |

`TOP_K_MATCH` checks the first divergent choice; it does not validate the rest of
the divergent continuation. DSpark matches same-batch target-only sequences
exactly on 29/40 prompts at batch 1 and 32/40 at batch 2. It verifies 3,395/3,297
draft tokens and accepts 830/842, respectively. Thus drafting is exercised, but
**bitwise serving losslessness is not established**. The exact lifecycle cases
above and the wider corpus's near-tie behavior are both part of qualification.

The reduced-precision **draft-forward** qualification failures below remain
unresolved. Different draft candidates can change acceptance and performance;
they do not bypass target verification. Serving output checks and checkpoint
candidate equivalence are separate requirements, and the experimental integration
does not establish complete DSpark qualification or a speedup.

### Full-continuation audit

Use `--audit-continuations` for a strict check that also covers every token after
the first divergence:

```bash
python -m tools.check_parity \
    --method dspark --num-draft-tokens 7 --dspark-draft-topk 64 \
    --model /path/to/target/snapshot --draft /path/to/draft/snapshot \
    --batch-size 1 4 --max-tokens 32 --audit-continuations \
    --output-dir /path/to/new-continuation-audit
```

The tool observes the actual target verification rows without requesting sample
logprobs, which would disable drafting. It then replays each emitted continuation
through native mlx-lm with fresh KV, forcing the observed tokens so that every
comparison uses the prefix serving actually produced. A native self-replay
control checks this replay against the free-running reference.

`continuation-audit.json` retains every position's emitted token, native argmax,
candidate rank and score gap, and the serving verifier's own argmax. It labels
prefill, ordinary decode, accepted draft, correction, and bonus decisions. The
summary separates mismatches against native MLX from mismatches against the
serving verifier; it also records exact free-running sequence counts. Checkpoint
paths are resolved once for all workers and recorded with source hashes and
package versions in `metadata.json`.

Any greedy mismatch, including a tie with a different argmax, fails the strict
audit and makes the command exit nonzero. Ranks and gaps are diagnostics, not
tolerances. Missing or invalid evidence also fails. The legacy `TOP_K_MATCH`
report cannot override these failures. This qualifies greedy decisions for the
tested workload; it does not establish sampled verification or measure speed.

## Measure HTTP serving and memory

Run the matched target-only/DSpark comparison from a source checkout on macOS:

```bash
python -m tools.benchmark.dspark_serving_benchmark \
    --target /path/to/Qwen3-4B-4bit/snapshot \
    --dspark /path/to/dspark_qwen3_4b_block7/snapshot \
    --concurrency 1 4 --repeats 2 --output-len 64 \
    --output-dir /path/to/new-benchmark-results
```

Add `--draft /path/to/Qwen3-0.6B/snapshot` for an ordinary draft-model arm.
All arms use the same target, tokenizer, plain greedy prompts, output length,
context/prefill limits, memory fraction, and disabled prefix caching.
The default workload is the shared 40-prompt corpus. For representative serving
measurements, supply natural prompts as `--prompt-file` JSONL `{"prompt": "..."}` rows and `--num-prompts`.
Prompts must fit the context limit without truncation.
Use `--dspark-draft-topk 64` for candidate limiting; draft widths are recorded
separately (`--dspark-width 7`, `--draft-width 3`).

Each arm runs in a fresh loopback-only server process group, with arm and
concurrency order reversed between repeats. Each timed `vllm bench serve` run
follows a discarded full-workload streaming warmup. Both use independent
concurrent requests, with identical input and output lengths in every arm.
No parity pass or inference instrumentation runs in the timing window.

`summary.json` contains throughput, latency, and ratios paired with the same
repeat's target-only baseline, including the median of those paired ratios.
Missing or duplicate pairs and mismatched workloads fail comparison.
Commands, detailed measurements, source hashes, server logs, and memory/counter
snapshots are retained. Counters exclude warmup and must show actual verified
drafts in speculative arms; incomplete requests or missing counters fail the run.

Benchmark success means the requested workload completed and its measurements
are valid. It does **not** qualify output parity. Run `tools.check_parity --method
dspark --top-k 5` separately with the same checkpoints and draft settings; add
`--method draft_model` in a separate run when measuring that arm. The former
untimed token pass and exact-match fields have been removed from the benchmark:
correctness reports now belong to the shared parity tool.

Worker snapshots distinguish MLX active/peak allocation, allocator cache, worker
RSS and lifetime peak RSS, and physical KV backing bytes. The startup snapshot
includes model loading and profiling; the measured phase resets only the MLX
peak counter. These are different accounting domains on unified memory and
must not be added together. Worker RSS excludes the API/client processes;
reserved KV bytes do not mean every page is resident. The scheduler's own
reported token capacity accounts for cache groups and is saved separately.
Equal memory fractions can yield different usable capacities across arms.

Follow the [macOS benchmark guide](benchmarking-macos.md) for warmup, power,
thermal state and desktop contention. Before/after machine observations are
saved; they cannot rule out interference during a run. Natural continuations
can differ even under greedy sampling, so these measurements do not establish
bitwise losslessness or a general speedup.

To reproduce the RFC's Qwen3-8B/0.6B workload, use these pinned local snapshots
in the same command:

- Target: [mlx-community/Qwen3-8B-4bit](https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192).
- DSpark: [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/tree/03326e5043815da1f81b109078b2889737c26017).
- Ordinary draft: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).

Compare the full-vocabulary and candidate-limited configurations in separate
output directories, keeping all other settings fixed. This pair uses more
memory than the 4B pair; record the memory fraction and cache capacity for
every arm.

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
- In the DeepSeek checkpoint, every block query attends the full committed prefix
  and the complete block. Speculators checkpoints use their configured prefix
  windows and within-block masks, as described above.
  `DSparkModel.block_hidden` returns normalized states for all K positions.
- `greedy_proposal` adds the vanilla low-rank Markov correction sequentially:
  the first position uses the anchor; later positions use the actual preceding
  proposal. It returns IDs, corrected logits, and optional raw confidence logits.
  Confidence uses that same predecessor and does not truncate the proposal.
- The loader accepts local, unsharded, uniform FP32/FP16/BF16 safetensors and
  preserves their precision. Tensor names/shapes and finite weights are checked.
  DFlash and DSpark share these checks in `draft_checkpoint.py`.
  Pre-quantized, gated/RNN-head, GIDD, scaled/partial-RoPE, and non-Qwen3 checkpoints
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
changes subsequent Markov corrections. BF16 token equivalence and FP16 checkpoint
qualification therefore remain unestablished. Serving output is checked separately
through target verification.

Small independent CPU-math tests cover block attention, Markov recurrence,
confidence predecessor alignment, optional heads, malformed checkpoints, and
compiled replay. Run them with `pytest tests/test_dspark.py`.

## Scheduler-owned draft KV

`DSparkPagedCache` binds the draft layers' views of a `KVCacheStorage` allocation.
It shares context projection, scatter, and block attention with `DFlashPagedCache`;
each adapter keeps its checkpoint's embeddings, prediction alignment, and heads.
The binding consumes scheduler block tables and never allocates request pages.
Bound layers must be distinct, belong to one scheduler group, and share the
cache block size and precision. Incompatible bindings are rejected before writes.

- `write_context(features, spans)` takes packed target features and
  `(block_ids, first_position, row_count)` spans. Commit only verified target rows.
  After verification, overwrite temporary block KV with those target-feature
  projections, including for accepted draft tokens.
- `compile_draft(num_draft_tokens=K)` returns a callable taking anchor IDs and
  `(block_ids, committed_length)` rows. It returns IDs, corrected logits, and
  optional raw confidence. It reserves **K slots**, including the anchor, and
  predicts from every slot. DFlash retains its separate K+1-slot contract.
- Full-attention layers expose the entire committed prefix and draft block.
  K=1 uses ordinary paged decode; larger blocks use explicit bidirectional ranges.
  Sliding layers gather the live prefix window and apply their configured block
  mask through SDPA.
- Fixed-size block tables and device offsets let a compiled block replay as
  context grows. Sliding-window gather shapes use bounded buckets. Cache writes
  retain dependencies through shared storage.
  The caller owns committed lengths, page ownership, and invalidation after
  preemption or request reuse; the cache does not track request identities.

`pytest tests/test_dspark_paged.py` exercises actual Metal kernels with FP16/BF16,
cache blocks 8/16/32, widths 1/3/7, optional confidence heads, ragged requests,
incremental verified commits, rejected tails, and reused pages. Independent CPU
attention checks logits and confidence. Exact-ID assertions use well-separated
Markov choices so numerical near ties do not obscure cache errors. Tests also
check page bounds, untouched target storage, and replay without retracing.

The full-checkpoint diagnostic compares paged proposals against native dense
DSpark at the same precision, using the snapshots above:

```bash
python -m tools.dspark_paged_parity \
    --target /path/to/target/snapshot \
    --draft /path/to/draft/snapshot \
    --dtype float16 \
    --output /path/to/new-paged-results.json
```

It records all 42 cases, source hashes, tensor errors, and exact-token checks,
and exits unsuccessfully if any gate fails. On the qualified pair, FP16 matches
all 231 proposal IDs, but six one-token-prefix cases exceed the strict final-logit
bound (`atol=0.015`, `rtol=0.02`; maximum absolute error 0.04004).
The bound is retained and these cases remain failures. This cache diagnostic does
not establish reduced-precision checkpoint equivalence or serving losslessness;
use the separate serving checks above to assess target-verified output.

`--dtype bfloat16` passes 41 of the 42 native comparisons, but fails at batch 2,
context 1, K=7: the dense path ties at 16.75, while paged attention produces
16.75 versus 16.625. The changed choice affects subsequent Markov corrections.
