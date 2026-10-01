# SPDX-License-Identifier: Apache-2.0
"""Real scheduler-driven DFlash width changes, pause/resume, and preemption."""

import json
import os

import pytest

from tests.test_dflash_serving_e2e import _serve


def _serve_scheduled(baseline_path, verify_window, pressure=False):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_METAL_SPEC_VERIFY_WINDOW"] = "1" if verify_window else "0"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="mlx-community/Qwen3-4B-4bit",
        max_model_len=128,
        max_num_seqs=3,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=10 if pressure else 14,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 2, 1], [3, 3, 0]],
        },
    )
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    runner = engine.model_executor.driver_worker.model_runner
    proposer = runner._drafter
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.encode("Explain how a computer works. " * 20)[:n] for n in (13, 63)
    ] + [tokenizer.encode("Describe why plants grow. " * 20)[:61]]
    reference = json.loads(baseline_path.read_text())["outputs"]
    expected = [reference[i][:32] for i in (0, 3, 4)]
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    widths, handoffs = [], set()
    stats = {"preemptions": 0, "resumptions": 0, "accepted": 0, "rejected": 0}
    propose = proposer.propose
    schedule = scheduler.schedule
    verify = runner._spec_decode_controller.verify_greedy

    def record_propose(ctx):
        result = propose(ctx)
        width = ctx.num_speculative_tokens
        widths.append(width)
        handoffs.update((len(s.draft_token_ids), width) for s in ctx.decode_segments)
        if width == 0:
            assert result is None
        elif result is not None:
            assert all(len(tokens) == width for tokens in result.draft_token_ids)
        return result

    def record_schedule(*args, **kwargs):
        result = schedule(*args, **kwargs)
        stats["preemptions"] += len(result.preempted_req_ids)
        stats["resumptions"] += len(result.scheduled_cached_reqs.resumed_req_ids)
        return result

    def record_verify(logits, requests, segments):
        result = verify(logits, requests, segments)
        for segment, tokens in zip(segments, result, strict=True):
            stats["accepted"] += len(tokens) - 1
            stats["rejected"] += len(segment.draft_token_ids) - len(tokens) + 1
        return result

    proposer.propose = record_propose
    scheduler.schedule = record_schedule
    runner._spec_decode_controller.verify_greedy = record_verify

    def step_until(predicate):
        outputs = {}
        for _ in range(500):
            if predicate():
                return outputs
            assert engine.has_unfinished_requests(), "Requests ended before transition"
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = list(output.outputs[0].token_ids)
        raise AssertionError("Scheduler did not reach the expected transition")

    def free_blocks():
        return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    try:
        free = free_blocks()
        if pressure:
            # Admit two long requests into a pool that cannot retain both as
            # they grow. Recompute must preserve features across K=1/K=3.
            for i, prompt in enumerate(prompts[1:]):
                engine.add_request(
                    str(i),
                    {"prompt_token_ids": prompt},
                    SamplingParams(temperature=0, max_tokens=48, ignore_eos=True),
                )
            outputs = step_until(lambda: not engine.has_unfinished_requests())
            assert [outputs[str(i)] for i in range(2)] == reference[3:]
            assert all(stats.values()), stats
            assert {1, 3} <= set(widths)
            assert free_blocks() == free
            print("DFlash scheduled pressure:", stats, flush=True)
            return
        engine.add_request("lead", {"prompt_token_ids": prompts[0]}, params)
        outputs = step_until(lambda: 3 in widths)
        engine.add_request(
            "second",
            {"prompt_token_ids": prompts[1]},
            SamplingParams(temperature=0, max_tokens=48, ignore_eos=True),
        )
        outputs.update(step_until(lambda: 1 in widths))
        cancelled = engine.add_request(
            "reused", {"prompt_token_ids": prompts[2]}, params
        )
        outputs.update(step_until(lambda: widths.count(0) >= 3))
        assert not outputs, "The requests must stay live across the width changes"
        engine.abort_request(["reused"])
        outputs.update(step_until(lambda: not engine.has_unfinished_requests()))
        assert outputs == {"lead": expected[0], "second": reference[3]}
        assert {(3, 1), (1, 0), (0, 0), (0, 1), (1, 3)} <= handoffs, handoffs
        assert set(proposer._drafts) == {1, 3}
        assert free_blocks() == free

        # Resume speculation on freshly allocated pages under a cancelled ID.
        engine.add_request("reused", {"prompt_token_ids": prompts[0]}, params)
        assert step_until(lambda: not engine.has_unfinished_requests()) == {
            "reused": expected[0]
        }
        assert cancelled not in proposer._valid_ends
        assert free_blocks() == free

        assert stats["accepted"] > 0 and stats["rejected"] > 0
        print(
            "DFlash dynamic schedule:", stats, "handoffs:", sorted(handoffs), flush=True
        )
    finally:
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("verify_window", [False, True])
def test_dflash_scheduler_width_transitions(
    tmp_path, run_in_spawn_process, verify_window
):
    baseline = tmp_path / "target.json"
    run_in_spawn_process(_serve, "target", baseline, verify_window)
    run_in_spawn_process(_serve_scheduled, baseline, verify_window)
    run_in_spawn_process(_serve_scheduled, baseline, verify_window, True)
