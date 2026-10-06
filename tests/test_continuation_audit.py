# SPDX-License-Identifier: Apache-2.0
"""Whole-continuation qualification must not hide a failure after a near tie."""

import copy
import importlib
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from tools.continuation_audit import (
    audit_outputs,
    decision_kinds,
    replay_tokens,
    score_token,
)


def scores(values, token):
    return score_token(np.array(values, dtype=np.float32), token)


def evidence(tokens):
    return {
        "input_ids": [4, 3],
        "tokens": tokens,
        "decisions": [scores(np.eye(6)[token], token) for token in tokens],
    }


@pytest.mark.parametrize("tokens,argmax", [([4], [1]), ([4, 5, 0], [1, 5, 0])])
def test_replay_uses_emitted_prefix_and_fresh_native_cache(tokens, argmax):
    class Model:
        def __init__(self):
            self.calls = []
            self.caches = 0
            # Free greedy: 0 -> 1 -> 2 -> 3. Observed: 0 -> 4 -> 5 -> 0.
            self.logits = mx.eye(6)[mx.array([1, 2, 3, 0, 5, 0])] * 10

        def make_cache(self):
            self.caches += 1
            return []

        def __call__(self, inputs, cache):
            self.calls.extend(inputs[0].tolist())
            return self.logits[inputs]

    model = Model()
    for _ in range(2):
        rows = replay_tokens(model, [0], tokens)
        assert [row["argmax"] for row in rows] == argmax
        assert [row["token"] for row in rows] == tokens
    assert model.caches == 2
    # mlx-lm also evaluates one final lookahead row, which is never reported.
    assert model.calls == ([0] + tokens) * 2


def test_later_failure_is_retained_after_first_divergent_tie():
    reference = evidence([0, 2, 3])
    output = evidence([1, 4, 5])
    replayed = [
        scores([0, 0, -1, -2, -3, -4], 1),  # tie still fails strict argmax
        scores([-5, -4, -3, -2, 0, -1], 4),  # correct on the divergent prefix
        scores([0, -1, -2, -3, -4, -9], 5),  # later error outside top five
    ]
    prefixes = []

    def replay(prompt, tokens):
        prefixes.append((prompt, tokens))
        return replayed

    report = audit_outputs([reference], [output], replay, max_tokens=3)
    assert prefixes == [([4, 3], [1, 4, 5])]
    assert not report["passed"]
    assert report["tokens"] == 3
    assert report["native_argmax_mismatches"] == 2
    assert report["native_tied_mismatches"] == 1
    assert report["native_outside_top5"] == 1
    assert report["max_native_argmax_gap"] == 9
    assert report["serving_argmax_mismatches"] == 0
    assert [row["prefix_length"] for row in report["results"][0]["rows"]] == [2, 3, 4]


def test_verifier_choice_and_native_choice_are_separate_checks():
    output = evidence([1, 2])
    output["decisions"][1] = scores([0, 9, 8, 0, 0, 0], 2)
    replayed = evidence([1, 2])["decisions"]
    report = audit_outputs([output], [output], lambda *_: replayed, max_tokens=2)
    assert not report["passed"]
    assert report["native_argmax_mismatches"] == 0
    assert report["serving_argmax_mismatches"] == 1


def test_complete_exact_control_passes_without_serving_evidence():
    row = evidence([1, 2])
    control = {key: row[key] for key in ("input_ids", "tokens")}
    report = audit_outputs(
        [control], [control], lambda *_: row["decisions"], max_tokens=2, serving=False
    )
    assert report["passed"] and report["exact_sequences"] == 1
    assert report["serving_argmax_mismatches"] is None


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing_sequence",
        "prompt",
        "truncated",
        "truncated_reference",
        "missing_decisions",
        "null_decisions",
        "null_serving_row",
        "null_native_row",
        "row_token",
        "bool_row_token",
        "missing_replay",
        "nan",
        "negative_gap",
        "bad_rank",
        "bool_token",
        "negative_token",
    ],
)
def test_incomplete_or_invalid_evidence_cannot_pass(problem):
    ref, got = evidence([1, 2]), evidence([1, 2])
    native = copy.deepcopy(ref["decisions"])
    refs, outputs = [ref], [got]
    if problem == "empty":
        refs, outputs = [], []
    elif problem == "missing_sequence":
        outputs = []
    elif problem == "prompt":
        got["input_ids"] = [99]
    elif problem == "truncated":
        got["tokens"].pop()
    elif problem == "truncated_reference":
        ref["tokens"].pop()
    elif problem == "missing_decisions":
        got["decisions"].pop()
    elif problem == "null_decisions":
        got["decisions"] = None
    elif problem == "null_serving_row":
        got["decisions"][0] = None
    elif problem == "null_native_row":
        native[0] = None
    elif problem == "row_token":
        native[1]["token"] = 3
    elif problem == "bool_row_token":
        native[0]["token"] = True
    elif problem == "missing_replay":
        native.pop()
    elif problem == "nan":
        native[1]["argmax_gap"] = float("nan")
    elif problem == "negative_gap":
        native[1]["argmax_gap"] = -1
    elif problem == "bad_rank":
        native[1]["rank"] = 0
    elif problem == "bool_token":
        got["tokens"][0] = True
    else:
        got["tokens"][0] = -1
    with pytest.raises(ValueError):
        audit_outputs(refs, outputs, lambda *_: native, max_tokens=2)


@pytest.mark.parametrize(
    "values,token",
    [
        ([], 0),
        ([[1.0]], 0),
        ([float("nan")], 0),
        ([float("inf")], 0),
        ([0.0], -1),
        ([0.0], 1),
        ([0.0], True),
    ],
)
def test_scores_require_a_finite_vector_and_valid_token(values, token):
    with pytest.raises(ValueError):
        scores(values, token)


def test_score_ranking_preserves_lowest_token_id_ties():
    row = scores([0, 1, 1, 0], 2)
    assert row["argmax"] == 1 and row["rank"] == 2 and row["argmax_gap"] == 0
    assert [item["id"] for item in row["top_logprobs"]] == [1, 2, 0, 3]


def test_report_normalization_does_not_change_the_verifiers_argmax():
    row = scores([0, 1e-9, 0], 1)
    assert row["argmax"] == 1 and row["rank"] == 1 and row["argmax_gap"] == 0
    assert row["top_logprobs"][0]["id"] == 1


@pytest.mark.parametrize(
    "drafts,emitted,kinds",
    [
        ([], [4], ["decode"]),
        ([1, 2], [4], ["correction"]),
        ([1, 2], [1, 4], ["accepted", "correction"]),
        ([1, 2], [1, 2, 4], ["accepted", "accepted", "bonus"]),
    ],
)
def test_verification_rows_identify_accepted_correction_and_bonus(
    drafts, emitted, kinds
):
    assert decision_kinds(drafts, emitted) == kinds


@pytest.mark.parametrize(
    "drafts,emitted",
    [
        ([], []),
        ([], [1, 2]),
        ([1, 2], [1]),
        ([1, 2], [3, 4]),
    ],
)
def test_invalid_verification_alignment_is_rejected(drafts, emitted):
    with pytest.raises(ValueError):
        decision_kinds(drafts, emitted)


@pytest.mark.parametrize("problem", ["truncated", "wrong_token", "extra_token"])
def test_replay_rejects_broken_generation_and_closes_stream(monkeypatch, problem):
    closed = []

    def generate(*args, **kwargs):
        try:
            yield 1, mx.array([0.0, 1.0, 0.0])
            if problem == "truncated":
                return
            yield (0 if problem == "wrong_token" else 2), mx.array([0.0, 0.0, 1.0])
            if problem == "extra_token":
                yield 0, mx.array([1.0, 0.0, 0.0])
        finally:
            closed.append(True)

    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"), "generate_step", generate
    )
    with pytest.raises(ValueError):
        replay_tokens(None, [0], [1, 2])
    assert closed == [True]


@pytest.mark.parametrize(
    "audit_passed,legacy_passed", [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("hub_models", [False, True])
def test_cli_requires_both_audits_and_preserves_the_report(
    monkeypatch, tmp_path, audit_passed, legacy_passed, hub_models
):
    import tools.dflash_serving_parity as tool

    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    output = tmp_path / "results"
    resolved = []

    def snapshot(model):
        resolved.append(model)
        return str(target if model == "test/target" else draft)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            "--method",
            "dspark",
            "--target",
            "test/target" if hub_models else str(target),
            "--draft",
            "test/draft" if hub_models else str(draft),
            "--audit-continuations",
            "--output-dir",
            str(output),
        ],
    )
    commands = []
    report = {
        "passed": audit_passed,
        "reports": {
            "dspark-b1": {
                "tokens": 32,
                "native_argmax_mismatches": int(not audit_passed),
                "serving_argmax_mismatches": 0,
            }
        },
    }
    real_run = tool.subprocess.run

    def run(command, **kwargs):
        if "--worker" not in command:
            # vLLM probes the OS while its schedule utility is imported.
            return real_run(command, **kwargs)
        commands.append(command)
        worker = command[-1]
        assert kwargs["check"]
        assert kwargs["timeout"] == (3000 if worker == "replay" else 600)
        if worker == "native":
            (output / "native.json").write_text("[]")
        elif worker == "replay":
            (output / "continuation-audit.json").write_text(json.dumps(report))
        else:
            for batch in (1, 2):
                (output / f"{worker}-b{batch}.json").write_text(
                    json.dumps({"stats": {}, "outputs": []})
                )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    # A legacy TOP_K_MATCH success must not override the strict audit result.
    monkeypatch.setattr(tool, "compare_results", lambda *a, **kw: legacy_passed)
    if audit_passed and legacy_passed:
        tool.main()
    else:
        with pytest.raises(SystemExit) as caught:
            tool.main()
        assert caught.value.code == 1
    assert json.loads((output / "continuation-audit.json").read_text()) == report
    assert [command[-1] for command in commands] == [
        "native",
        "target",
        "dspark",
        "replay",
    ]
    assert all(
        command[-6:-2] == ["--target", str(target), "--draft", str(draft)]
        for command in commands
    )
    assert resolved == (["test/target", "test/draft"] if hub_models else [])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["target"] == str(target)
    assert metadata["draft"] == str(draft)


def test_cli_rejects_duplicate_batch_sizes_before_starting_workers(
    monkeypatch, tmp_path
):
    import tools.dflash_serving_parity as tool

    monkeypatch.setattr(
        "sys.argv",
        ["parity", "--batch-size", "1", "1", "--output-dir", str(tmp_path / "results")],
    )
    with pytest.raises(SystemExit) as caught:
        tool.main()
    assert caught.value.code == 2
    assert not (tmp_path / "results").exists()
