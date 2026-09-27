# SPDX-License-Identifier: Apache-2.0
"""Exercise the public statistics RPC with in-process and spawned engines."""

import multiprocessing as mp
import os

import pytest


def _run_stats_rpc(engine_multiprocessing):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = str(int(engine_multiprocessing))
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=10,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "max_model_len": 64,
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 2, 1]],
        },
    )
    try:
        expected = {
            "num_context_limit_fallback_requests": 0,
            "min_draft_tokens": 1,
            "max_model_len": 64,
        }
        assert llm.collective_rpc("get_draft_model_stats", timeout=60) == [expected]
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.encode("Explain how a computer works. " * 20)[:55],
            tokenizer.encode("Describe why plants grow. " * 20)[:53],
        ]
        params = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        for count in (2, 4):
            results = llm.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                params,
                use_tqdm=False,
            )
            assert all(len(result.outputs[0].token_ids) == 48 for result in results)
            expected["num_context_limit_fallback_requests"] = count
            snapshot = llm.collective_rpc("get_draft_model_stats", timeout=60)
            assert snapshot == [expected]
            snapshot[0]["num_context_limit_fallback_requests"] = -1
            assert llm.collective_rpc("get_draft_model_stats", timeout=60) == [expected]
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("engine_multiprocessing", [False, True])
def test_draft_model_stats_rpc_e2e(engine_multiprocessing):
    process = mp.get_context("spawn").Process(
        target=_run_stats_rpc, args=(engine_multiprocessing,)
    )
    process.start()
    try:
        process.join(timeout=300)
        assert not process.is_alive(), "Draft statistics RPC timed out"
        assert process.exitcode == 0, "Draft statistics RPC serving check failed"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
