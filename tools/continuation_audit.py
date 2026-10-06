# SPDX-License-Identifier: Apache-2.0
"""Audit greedy choices on every emitted prefix, including divergent tails.

Native replay forces the observed tokens through mlx-lm's normal generation
loop. Its scores therefore describe the prefix that serving actually used,
even after serving diverges from the native free-running sequence. Candidate
ranks and score gaps are diagnostics; only exact argmax agreement passes.
"""

from __future__ import annotations

import json
from numbers import Integral

import numpy as np


def token_ids(values, *, name: str) -> list[int]:
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in values
        )
    ):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return [int(value) for value in values]


def score_token(scores: np.ndarray, token: int) -> dict:
    """Describe a token under one finite vocabulary row, with stable ties."""
    scores = np.asarray(scores)
    token = token_ids([token], name="emitted token")[0]
    if (
        scores.ndim != 1
        or not scores.size
        or not np.issubdtype(scores.dtype, np.floating)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Expected a nonempty, finite vocabulary score vector")
    if token >= scores.size:
        raise ValueError(f"Token {token} is outside vocabulary size {scores.size}")
    best = int(np.argmax(scores))
    selected = float(scores[token])
    indices = np.argsort(-scores, kind="stable")[:5]
    # Decide using the input scores. Normalizing low-magnitude logits in FP32
    # can turn distinct values into a tie; logprobs are only for the report.
    shifted = scores.astype(np.float64) - float(scores[best])
    logprobs = shifted - np.log(np.exp(shifted).sum())
    # Include tied lower IDs, matching argmax's first-index tie break.
    rank = 1 + int(np.count_nonzero(scores > selected))
    rank += int(np.count_nonzero(scores[:token] == selected))
    return {
        "token": token,
        "argmax": best,
        "rank": rank,
        "argmax_gap": float(scores[best]) - selected,
        "top_logprobs": [
            {"id": int(i), "text": "", "rank": j, "logprob": float(logprobs[i])}
            for j, i in enumerate(indices, start=1)
        ],
    }


def decision_kinds(drafts: list[int], emitted: list[int]) -> list[str]:
    """Label the accepted prefix and its correction or bonus before output trimming."""
    if not emitted or len(emitted) > len(drafts) + 1:
        raise ValueError("Invalid number of greedy verification outputs")
    if not drafts:
        return ["decode"]
    accepted = len(emitted) - 1
    if emitted[:accepted] != drafts[:accepted]:
        raise ValueError("Accepted output does not match the draft prefix")
    if accepted < len(drafts) and emitted[-1] == drafts[accepted]:
        raise ValueError("Correction did not reject the next draft token")
    return ["accepted"] * accepted + [
        "bonus" if accepted == len(drafts) else "correction"
    ]


def replay_tokens(model, input_ids: list[int], tokens: list[int]) -> list[dict]:
    """Score an observed continuation using native one-token decode and fresh KV."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    input_ids = token_ids(input_ids, name="prompt")
    tokens = token_ids(tokens, name="continuation")
    position = 0

    def force_token(logprobs):
        nonlocal position
        # generate_step evaluates one extra lookahead sample before its final
        # yield. That sample is not audited or appended to the continuation.
        if position > len(tokens):
            raise ValueError("Native generation sampled beyond its lookahead")
        token = tokens[min(position, len(tokens) - 1)]
        if token >= logprobs.shape[-1]:
            raise ValueError(f"Forced token {token} is outside the native vocabulary")
        position += 1
        return mx.array([token], dtype=mx.uint32)

    stream = generate_step(
        mx.array(input_ids, dtype=mx.uint32),
        model,
        max_tokens=len(tokens),
        sampler=force_token,
    )
    rows = []
    try:
        for i, (token, logprobs) in enumerate(stream):
            if i >= len(tokens) or token != tokens[i]:
                raise ValueError(
                    "Native replay did not follow the emitted continuation"
                )
            rows.append(score_token(np.array(logprobs.astype(mx.float32)), token))
    finally:
        stream.close()
    if len(rows) != len(tokens):
        raise ValueError("Native replay returned an incomplete continuation")
    return rows


def audit_outputs(
    reference: list[dict],
    outputs: list[dict],
    replay,
    *,
    max_tokens: int,
    serving: bool = True,
) -> dict:
    """Fail closed on incomplete evidence and retain every token comparison."""
    if not reference or len(reference) != len(outputs) or max_tokens < 1:
        raise ValueError("Audit needs complete, nonempty prompt/output pairs")
    sequences = []
    for ref, output in zip(reference, outputs, strict=True):
        prompt = token_ids(ref["input_ids"], name="reference prompt")
        if token_ids(output["input_ids"], name="serving prompt") != prompt:
            raise ValueError("Serving and native input token IDs differ")
        expected = token_ids(ref["tokens"], name="native continuation")
        tokens = token_ids(output["tokens"], name="serving continuation")
        if len(tokens) != max_tokens or len(expected) != max_tokens:
            raise ValueError("Audit requires the complete requested continuation")
        recorded = output["decisions"] if serving else None
        if serving and (not isinstance(recorded, list) or len(recorded) != max_tokens):
            raise ValueError("Missing serving decision rows")
        native = replay(prompt, tokens)
        if len(native) != max_tokens:
            raise ValueError("Missing native replay rows")
        rows = []
        for i, (token, native_row) in enumerate(zip(tokens, native, strict=True)):
            serving_row = recorded[i] if recorded is not None else None
            required = (native_row, serving_row) if serving else (native_row,)
            for row in required:
                if not isinstance(row, dict):
                    raise ValueError("Missing greedy-decision evidence")
                if type(row["token"]) is not int or row["token"] != token:
                    raise ValueError(
                        "A decision row describes a different emitted token"
                    )
                if (
                    type(row["argmax"]) is not int
                    or row["argmax"] < 0
                    or type(row["rank"]) is not int
                    or row["rank"] < 1
                    or not np.isfinite(row["argmax_gap"])
                    or row["argmax_gap"] < 0
                    or ((row["argmax"] == token) != (row["rank"] == 1))
                    or (row["argmax"] == token and row["argmax_gap"] != 0)
                ):
                    raise ValueError("Invalid greedy-decision evidence")
            rows.append(
                {
                    "position": i,
                    "prefix_length": len(prompt) + i,
                    "native": native_row,
                    "serving": serving_row,
                    "native_argmax_match": token == native_row["argmax"],
                    "serving_argmax_match": token == serving_row["argmax"]
                    if serving_row is not None
                    else None,
                }
            )
        sequences.append(
            {
                "input_ids": prompt,
                "tokens": tokens,
                "exact_sequence": tokens == expected,
                "rows": rows,
            }
        )
    rows = [row for sequence in sequences for row in sequence["rows"]]
    native_mismatches = sum(not row["native_argmax_match"] for row in rows)
    serving_mismatches = sum(row["serving_argmax_match"] is False for row in rows)
    exact_sequences = sum(sequence["exact_sequence"] for sequence in sequences)
    return {
        "passed": native_mismatches == serving_mismatches == 0
        and exact_sequences == len(sequences),
        "sequences": len(sequences),
        "tokens": len(rows),
        "exact_sequences": exact_sequences,
        "native_argmax_mismatches": native_mismatches,
        "native_tied_mismatches": sum(
            not row["native_argmax_match"] and row["native"]["argmax_gap"] == 0
            for row in rows
        ),
        "native_outside_top5": sum(row["native"]["rank"] > 5 for row in rows),
        "max_native_argmax_gap": max(row["native"]["argmax_gap"] for row in rows),
        "serving_argmax_mismatches": serving_mismatches if serving else None,
        "results": sequences,
    }


def audit_serving_run(args) -> dict:
    from mlx_lm import load

    from vllm_metal.compat import _patch_transformers_exaone4_config

    _patch_transformers_exaone4_config()
    model, _ = load(args.target)
    reference = json.loads((args.output_dir / "native.json").read_text())

    def replay(prompt, tokens):
        return replay_tokens(model, prompt, tokens)

    reports = {
        "native_control": audit_outputs(
            reference, reference, replay, max_tokens=args.max_tokens, serving=False
        )
    }
    for worker in ("target", args.method):
        for batch_size in args.batch_size:
            data = json.loads(
                (args.output_dir / f"{worker}-b{batch_size}.json").read_text()
            )
            reports[f"{worker}-b{batch_size}"] = audit_outputs(
                reference, data["outputs"], replay, max_tokens=args.max_tokens
            )
    return {
        "scope": "Exact greedy choices on every emitted prefix; ranks and gaps are diagnostics, not tolerances. This does not qualify sampled verification.",
        "passed": all(report["passed"] for report in reports.values()),
        "reports": reports,
    }
