# SPDX-License-Identifier: Apache-2.0
"""Shared exact and two-sided top-k comparison for serving parity tools."""

from __future__ import annotations

import math
from itertools import zip_longest


def _top_candidates(row: dict, position: int, top_k: int) -> dict:
    steps = row["top_logprobs"]
    if position >= len(steps):
        raise ValueError("Missing logprobs at the first divergent token")
    candidates = {}
    for entry in steps[position]:
        rank = entry["rank"]
        if rank is None or rank > top_k:
            continue
        if (
            type(rank) is not int
            or rank < 1
            or entry["id"] in candidates
            or not math.isfinite(entry["logprob"])
        ):
            raise ValueError("Invalid top-k candidate evidence")
        candidates[entry["id"]] = entry
    if not candidates:
        raise ValueError("Missing top-k candidate evidence")
    return candidates


def parity_report(
    reference: list[dict],
    outputs: list[dict],
    *,
    max_tokens: int,
    top_k: int | None = None,
    labels: tuple[str, str] = ("mlx-lm", "metal"),
) -> dict:
    """Report exact tokens or mutual top-k membership at the first divergence."""
    if not reference or len(reference) != len(outputs):
        raise ValueError("Parity needs complete, nonempty prompt/output pairs")
    if max_tokens < 1 or (top_k is not None and top_k < 1):
        raise ValueError("Token and top-k limits must be positive")
    results = []
    counts = dict.fromkeys(("EXACT", "TOP_K_MATCH", "FAIL"), 0)
    for index, (ref, got) in enumerate(zip(reference, outputs, strict=True)):
        if "input_ids" in ref and "input_ids" in got:
            if ref["input_ids"] != got["input_ids"]:
                raise ValueError("Parity inputs have different token IDs")
        mismatch = next(
            (
                i
                for i, (a, b) in enumerate(zip_longest(ref["tokens"], got["tokens"]))
                if a != b
            ),
            None,
        )
        complete = len(ref["tokens"]) == len(got["tokens"]) == max_tokens
        exact = complete and mismatch is None
        compatible = False
        ref_top, got_top = {}, {}
        if complete and mismatch is not None and top_k is not None:
            ref_top = _top_candidates(ref, mismatch, top_k)
            got_top = _top_candidates(got, mismatch, top_k)
            compatible = (
                ref["tokens"][mismatch] in got_top
                and got["tokens"][mismatch] in ref_top
            )
        status = "EXACT" if exact else "TOP_K_MATCH" if compatible else "FAIL"
        counts[status] += 1
        results.append(
            {"prompt_index": index, "status": status, "first_divergence": mismatch}
        )
        print(f"{status} {ref.get('prompt', index)!r}")
        if not exact:
            if not complete:
                print(
                    f"  Expected {max_tokens} tokens; got "
                    f"{len(ref['tokens'])} / {len(got['tokens'])}"
                )
            if mismatch is not None:
                print(f"  First differing token (0-based): {mismatch}")
            for label, row in zip(labels, (ref, got), strict=True):
                print(f"  {label}: {row['tokens']}\n          {row['text']!r}")
            if ref_top and got_top:
                candidates = (ref["tokens"][mismatch], got["tokens"][mismatch])
                for label, top in zip(labels, (ref_top, got_top), strict=True):
                    for token in candidates:
                        entry = top.get(token)
                        if entry is None:
                            print(f"  {label}: token {token} is outside top-{top_k}")
                        else:
                            print(
                                f"  {label}: token {token} {entry['text']!r}, "
                                f"rank {entry['rank']}, logprob {entry['logprob']:.6f}"
                            )
                print(
                    f"  Matching prefix length: {mismatch}; "
                    "remaining continuation not compared."
                )
    return {"passed": counts["FAIL"] == 0, "counts": counts, "results": results}


def compare_results(reference, outputs, *, max_tokens, top_k=None) -> bool:
    """Boolean interface retained for existing parity callers."""
    return parity_report(reference, outputs, max_tokens=max_tokens, top_k=top_k)[
        "passed"
    ]
