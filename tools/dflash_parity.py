# SPDX-License-Identifier: Apache-2.0
"""Qualify DFlash capture and block logits against a local official MLX reference.

This compares model forwards, not serving throughput or generated sequences.
Pass local target/draft snapshots and model_mlx.py from the revision in the docs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from tools.attention_bench_utils import package_versions
from vllm_metal.v1.dflash import DFlashTargetCapture, load_dflash


def compare(actual: mx.array, expected: mx.array) -> dict:
    actual, expected = (
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
    )
    if actual.shape != expected.shape or not actual.size:
        raise ValueError(f"Incomplete comparison: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite output in DFlash comparison")
    return {
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "exact": bool(np.array_equal(actual, expected)),
    }


def qualify(target_path: Path, draft_path: Path, reference_path: Path) -> dict:
    # Loading arbitrary remote Python is deliberately not part of this tool.
    spec = importlib.util.spec_from_file_location("dflash_reference", reference_path)
    if spec is None or spec.loader is None:
        raise ValueError("Reference must be a local model_mlx.py file")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    draft = load_dflash(
        draft_path,
        target_config=json.loads((target_path / "config.json").read_text()),
    )
    target, tokenizer = load(str(target_path))
    embed = target.model.embed_tokens
    project = embed.as_linear if target.args.tie_word_embeddings else target.lm_head
    # Read the reference configuration independently from the checkpoint JSON.
    raw = json.loads((draft_path / "config.json").read_text())
    values = {
        field.name: raw[field.name]
        for field in fields(reference.DFlashConfig)
        if field.name in raw
    }
    values.update(raw["dflash_config"])
    ref_draft = reference.DFlashDraftModel(reference.DFlashConfig(**values))
    ref_draft.load_weights(
        list(mx.load(next(draft_path.glob("*.safetensors"))).items()), strict=True
    )
    ref_draft.eval()
    ref_draft.bind(target)
    mx.eval(ref_draft.parameters())
    capture = DFlashTargetCapture(target, draft.config)
    prompts = [
        "Explain how a computer works in simple terms.",
        "Write a Python function that adds two numbers and explain it.",
    ]
    rows = []
    for batch in (1, 2):
        for length in (17, 33, 65):
            token_rows = [
                tokenizer.encode(prompts[i] * 12)[:length] for i in range(batch)
            ]
            if any(len(row) != length for row in token_rows):
                raise ValueError("Prompt did not produce the requested context length")
            tokens = mx.array(token_rows)
            native_logits, features = capture.run(target, tokens)
            original_layers = list(target.model.layers)
            try:
                reference._patch_model(target, ref_draft.config.target_layer_ids)
                ref_logits = target(tokens)
                # The MLX hook observes pre-norm decoder outputs. Match the HF
                # hidden_states[lid + 1] contract used by the PyTorch reference
                # when a checkpoint requests the final target layer.
                ref_features = tuple(
                    target.model.norm(feature)
                    if lid == ref_draft.config.num_target_layers - 1
                    else feature
                    for lid, feature in zip(
                        ref_draft.config.target_layer_ids,
                        target._hidden_states,
                        strict=True,
                    )
                )
                compare(native_logits, ref_logits)
                capture_checks = [
                    compare(a, b) for a, b in zip(features, ref_features, strict=True)
                ]
            finally:
                target.model.layers[:] = original_layers
                if hasattr(target, "_hidden_states"):
                    delattr(target, "_hidden_states")
            anchors = mx.argmax(native_logits[:, -1], axis=-1)
            draft.validate_anchors(anchors)
            for width in sorted(
                {2, min(5, draft.config.block_size), draft.config.block_size}
            ):
                forward = partial(
                    draft.draft_logits,
                    num_draft_tokens=width - 1,
                    embed=embed,
                    project=project,
                )
                actual = forward(anchors, features)
                compiled = mx.compile(forward)(anchors, features)
                block = mx.concatenate(
                    [
                        anchors[:, None],
                        mx.full(
                            (batch, width - 1),
                            draft.config.mask_token_id,
                            dtype=anchors.dtype,
                        ),
                    ],
                    axis=1,
                )
                expected = ref_draft(
                    block,
                    mx.concatenate(ref_features, axis=-1),
                    ref_draft.make_cache(),
                    logits_start=1,
                )
                comparison = compare(actual, expected)
                compiled_comparison = compare(compiled, expected)
                for logits in (actual, compiled):
                    np.testing.assert_array_equal(
                        np.array(mx.argmax(logits, axis=-1)),
                        np.array(mx.argmax(expected, axis=-1)),
                    )
                rows.append(
                    {
                        "batch": batch,
                        "context_length": length,
                        "block_size": width,
                        "proposal_positions": batch * (width - 1),
                        "logits": comparison,
                        "compiled_logits": compiled_comparison,
                        "capture": capture_checks,
                        "argmax_exact": True,
                    }
                )
                print(
                    f"PASS B={batch} context={length} block={width} "
                    f"eager_error={comparison['max_abs_error']} "
                    f"compiled_error={compiled_comparison['max_abs_error']}",
                    flush=True,
                )
    return {
        "target": str(target_path.resolve()),
        "draft": str(draft_path.resolve()),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "native_source_sha256": hashlib.sha256(
            Path(load_dflash.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "versions": package_versions("mlx", "mlx-lm", "numpy"),
        "capture_layer_ids": draft.config.capture_layer_ids,
        "draft_layers": draft.config.num_hidden_layers,
        "reference_final_norm_applied": draft.config.num_target_layers - 1
        in draft.config.target_layer_ids,
        "cases": rows,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "--output must name a new file, so a failed run cannot leave a stale pass"
        )
    result = qualify(args.target, args.draft, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
