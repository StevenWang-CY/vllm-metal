# SPDX-License-Identifier: Apache-2.0
"""Qualify the DSpark forward against a local official DeepSpec checkout.

Uses native mlx-lm target features and greedy anchors. This is model-forward
qualification, not generated-sequence parity or a serving performance claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
from functools import partial
from pathlib import Path

# Configure only CLI execution, before importing MLX or the target loader.
# Importing comparison helpers must not alter the caller's environment.
if __name__ == "__main__":
    os.environ["MLX_ENABLE_TF32"] = "0"

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load
from safetensors.torch import load_file
from transformers import Qwen3Config

from tools.attention_bench_utils import native_source_hashes, package_versions
from vllm_metal.v1.dflash import DFlashTargetCapture
from vllm_metal.v1.draft_checkpoint import load_draft_weights
from vllm_metal.v1.dspark import DSparkConfig, load_dspark

# Every function that decides which weights produce the numbers in a report.
NATIVE_SOURCES = (load_dspark, load_draft_weights, DFlashTargetCapture.run)


def compare(actual, expected, *, atol, rtol):
    actual = np.array(actual.astype(mx.float32))
    expected = expected.detach().float().cpu().numpy()
    if actual.shape != expected.shape or not actual.size:
        raise ValueError(f"Incomplete comparison: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite DSpark output")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
    }


def check_tokens(actual, expected, actual_logits, expected_logits):
    actual, expected = np.array(actual), expected.detach().cpu().numpy()
    if actual.shape != expected.shape:
        raise AssertionError("DSpark proposal shapes differ")
    mismatches = np.argwhere(actual != expected)
    if mismatches.size:
        row, position = mismatches[0]
        a, b = int(actual[row, position]), int(expected[row, position])
        native = np.array(actual_logits[row, position].astype(mx.float32))
        reference = expected_logits[row, position].detach().float().cpu().numpy()
        raise AssertionError(
            f"DSpark proposal mismatch at row {row}, position {position}: "
            f"native token {a}, reference token {b}; "
            f"native logits ({native[a]}, {native[b]}), "
            f"reference logits ({reference[a]}, {reference[b]})"
        )


def capture_samples(target_path, backbone, context_lengths):
    """Capture native target features/anchors, then release the target weights."""
    target, tokenizer = load(str(target_path))
    capture = DFlashTargetCapture(target, backbone)
    samples = []
    prompts = [
        "Explain how a computer works in simple terms. ",
        "Write a Python function that adds two numbers. ",
    ]
    for batch in (1, 2):
        for length in context_lengths:
            ids = [
                tokenizer.encode(prompts[i] * (length + 1))[:length]
                for i in range(batch)
            ]
            if any(len(row) != length for row in ids):
                raise ValueError("Prompt is too short for the requested context")
            logits, features = capture.run(target, mx.array(ids))
            anchors = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(anchors, features)
            samples.append(
                (
                    batch,
                    length,
                    np.array(anchors),
                    [np.array(f.astype(mx.float32)) for f in features],
                )
            )
    del target, tokenizer, capture, logits, features, anchors
    gc.collect()
    mx.clear_cache()

    return samples


def qualify(args):
    raw = json.loads((args.draft / "config.json").read_text())
    config = DSparkConfig.from_dict(raw)
    target_config = json.loads((args.target / "config.json").read_text())
    config.backbone.validate_target(target_config)
    samples = capture_samples(args.target, config.backbone, args.context_lengths)

    reference_root = args.reference.resolve()
    reference_file = reference_root / "deepspec/modeling/dspark/qwen3/modeling.py"
    if not reference_file.is_file():
        raise ValueError("--reference must be a local DeepSpec checkout")
    sys.path.insert(0, str(reference_root))
    module = importlib.import_module("deepspec.modeling.dspark.qwen3.modeling")
    if Path(module.__file__).resolve() != reference_file:
        raise ValueError("Imported DeepSpec does not match the requested reference")
    torch_config = Qwen3Config.from_dict(raw)
    torch_config._attn_implementation = "eager"
    with torch.device("meta"):
        reference = module.Qwen3DSparkModel(torch_config)
    # Assign avoids first allocating randomly initialized checkpoint-sized tensors.
    reference.load_state_dict(
        load_file(str(args.draft / "model.safetensors")), strict=True, assign=True
    )
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    mlx_dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    reference = reference.to(dtype=torch_dtype).eval()
    # The nonpersistent RoPE buffer was created on meta as well.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    reference.rotary_emb = Qwen3RotaryEmbedding(torch_config)
    draft = load_dspark(args.draft, target_config=target_config)
    draft.set_dtype(mlx_dtype)
    mx.eval(draft.parameters())
    widths = sorted({1, min(3, config.backbone.block_size), config.backbone.block_size})
    compiled = {
        width: mx.compile(partial(draft.draft, num_draft_tokens=width))
        for width in widths
    }
    atol, rtol = (1e-3, 1e-3) if args.dtype == "float32" else (0.25, 0.02)
    rows = []
    for batch, length, anchor_values, feature_values in samples:
        anchors = mx.array(anchor_values)
        features = [mx.array(f).astype(mlx_dtype) for f in feature_values]
        torch_features = torch.from_numpy(np.concatenate(feature_values, axis=-1)).to(
            torch_dtype
        )
        torch_anchors = torch.from_numpy(anchor_values.astype(np.int64))
        for width in widths:
            ids = torch.cat(
                [
                    torch_anchors[:, None],
                    torch.full((batch, width - 1), config.backbone.mask_token_id),
                ],
                dim=1,
            )
            with torch.inference_mode():
                hidden = reference._forward_backbone(
                    noise_embedding=reference.embed_tokens(ids),
                    target_hidden_states=torch_features,
                    position_ids=torch.arange(length + width)[None].expand(batch, -1),
                    attention_mask=None,
                    use_cache=False,
                    is_causal=False,
                )
                tokens, logits = reference.sample_draft_tokens(
                    reference.compute_logits(hidden),
                    first_prev_token_ids=torch_anchors,
                    hidden_states=hidden,
                    temperature=0,
                )
                previous = torch.cat([torch_anchors[:, None], tokens[:, :-1]], dim=1)
                confidence = reference.predict_confidence_step(
                    hidden, prev_token_ids=previous
                )
            checks = {
                "hidden": compare(
                    draft.block_hidden(anchors, features, num_draft_tokens=width),
                    hidden,
                    atol=atol,
                    rtol=rtol,
                )
            }
            for name, result in (
                ("eager", draft.draft(anchors, features, num_draft_tokens=width)),
                ("compiled", compiled[width](anchors, features)),
            ):
                actual_tokens, actual_logits, actual_confidence = result
                check_tokens(actual_tokens, tokens, actual_logits, logits)
                checks[name] = compare(actual_logits, logits, atol=atol, rtol=rtol)
                if confidence is None:
                    if actual_confidence is not None:
                        raise AssertionError("Unexpected confidence output")
                else:
                    checks[name + "_confidence"] = compare(
                        actual_confidence, confidence, atol=atol, rtol=rtol
                    )
            rows.append(
                {
                    "batch": batch,
                    "context_length": length,
                    "width": width,
                    "tokens_exact": True,
                    "checks": checks,
                }
            )
            print(f"PASS B={batch} context={length} K={width}", flush=True)
    sources = [
        "deepspec/modeling/dspark/qwen3/modeling.py",
        "deepspec/modeling/dspark/markov_head.py",
        "deepspec/modeling/dspark/common.py",
        "deepspec/utils/sampling.py",
    ]
    return {
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "reference": str(reference_root),
        "dtype": args.dtype,
        "atol": atol,
        "rtol": rtol,
        "reference_sha256": {
            name: hashlib.sha256((reference_root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "native_source_sha256": native_source_hashes(*NATIVE_SOURCES),
        "versions": package_versions("mlx", "mlx-lm", "torch", "transformers"),
        "cases": rows,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "draft", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[17, 33, 65, 257]
    )
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.context_lengths
        or min(args.context_lengths) < 1
    ):
        parser.error("Use a new output path and positive context lengths")
    torch.set_num_threads(8)
    result = qualify(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
