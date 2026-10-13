"""Diagnostic only: identical-arithmetic head locality, never a serving claim."""

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from tools.attention_bench_utils import attention_tolerances, ref_paged_attn
from vllm_metal.metal import get_ops

root = Path(__file__).resolve().parent
out = root / "window-shared-head-probe"
out.mkdir(exist_ok=False)
ops = get_ops()
protocol = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "patch": subprocess.check_output(["git", "diff"], text=True),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cores": ops.detected_gpu_core_count(),
    "dtypes": ["float16", "bfloat16"],
    "batches": [1, 4],
    "prefixes": [256, 768, 2100, 2800, 3500, 8192],
    "width": 8,
    "heads_per_group": [1, 2],
    "repeats": 5,
    "warmups": 10,
    "samples": 100,
    "order": "alternate original/shared and shared/original",
    "scope": "isolated synchronized attention calls, no serving or end-to-end claim",
}
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
results = []
for dtype in (mx.float16, mx.bfloat16):
    for batch in protocol["batches"]:
        for prefix in protocol["prefixes"]:
            mx.random.seed(825 + len(results))
            width = protocol["width"]
            length = prefix + width
            blocks_per_seq = (length + 15) // 16
            blocks = batch * blocks_per_seq
            q = mx.random.normal((batch * width, 32, 128)).astype(dtype)
            k = mx.random.normal((blocks, 16, 8, 128)).astype(dtype)
            v = mx.random.normal(k.shape).astype(dtype)
            rng = np.random.default_rng(825 + len(results))
            tables_np = rng.permutation(blocks).reshape(batch, blocks_per_seq).astype(np.int32)
            tables = mx.array(tables_np)
            lens = mx.array([length] * batch, dtype=mx.int32)
            cu = mx.array([i * width for i in range(batch + 1)], dtype=mx.int32)
            mx.eval(q, k, v, tables, lens, cu)

            def run():
                result = mx.array(0)
                ops.paged_attention_primitive(
                    q, k, v, 8, 128**-0.5, 0.0, tables, lens, cu,
                    16, length, -1, result, window_seqlen_q=width,
                )
                mx.eval(result)
                return result

            oracle = ref_paged_attn(
                query=q, key_cache=k, value_cache=v, query_lens=[width] * batch,
                kv_lens=[length] * batch, block_tables=tables_np, scale=128**-0.5,
            )
            oracle_np = np.array(oracle.astype(mx.float32))
            arrays, checks = {}, {}
            prior = ops._set_paged_dispatch_diagnostics(True)
            for heads in protocol["heads_per_group"]:
                ops._set_window_test_heads(heads)
                try:
                    y = run()
                    a = np.array(y.astype(mx.float32))
                    atol, rtol = attention_tolerances(dtype)
                    np.testing.assert_allclose(a, oracle_np, atol=atol, rtol=rtol)
                    if heads != 1:
                        np.testing.assert_array_equal(a.view(np.uint32), arrays[1].view(np.uint32))
                    arrays[heads] = a
                    checks[heads] = {"bitwise": True, "reference": True, "route": ops.last_paged_dispatch()}
                except Exception as exc:
                    checks[heads] = {"error": str(exc)}
            ops._set_paged_dispatch_diagnostics(False)
            pairs = []
            for order in ([1, 2], [2, 1], [1, 2], [2, 1], [1, 2]):
                timing = {}
                for heads in order:
                    if heads not in arrays:
                        continue
                    ops._set_window_test_heads(heads)
                    for _ in range(protocol["warmups"]):
                        run()
                    samples = []
                    for _ in range(protocol["samples"]):
                        start = time.perf_counter_ns()
                        run()
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    timing[heads] = {"median_ms": statistics.median(samples), "samples_ms": samples}
                pairs.append({"order": order, "timing": timing})
            ratios = {
                heads: [pair["timing"][1]["median_ms"] / pair["timing"][heads]["median_ms"] for pair in pairs]
                for heads in arrays if heads != 1
            }
            row = {
                "dtype": str(dtype), "batch": batch, "prefix": prefix,
                "checks": checks, "median_ratios": {h: statistics.median(r) for h, r in ratios.items()},
                "ranges": {h: [min(r), max(r)] for h, r in ratios.items()}, "pairs": pairs,
            }
            results.append(row)
            np.savez_compressed(out / f"outputs-{len(results):02d}.npz", **{str(h): a for h, a in arrays.items()})
            (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "pairs"}), flush=True)
            ops._set_paged_dispatch_diagnostics(prior)
            ops._set_window_test_heads(1)
            mx.clear_cache()
(out / "complete.json").write_text(json.dumps({"cases": len(results)}) + "\n")
