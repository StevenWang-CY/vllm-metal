# SPDX-License-Identifier: Apache-2.0
"""Numerical evidence must fail on missing data, non-finite values, or wrong IDs."""

import mlx.core as mx
import pytest
import torch

from tools.dspark_parity import check_tokens, compare


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([], []),
        ([[1]], [[1, 2]]),
        ([[float("nan")]], [[float("nan")]]),
        ([[float("inf")]], [[float("inf")]]),
        ([[1]], [[2]]),
    ],
)
def test_compare_rejects_invalid_evidence(actual, expected):
    with pytest.raises((ValueError, AssertionError)):
        compare(mx.array(actual), torch.tensor(expected), atol=1e-3, rtol=1e-3)


def test_compare_distinguishes_exact_from_tolerant_results():
    assert compare(mx.array([[1.0]]), torch.tensor([[1.0]]), atol=1e-3, rtol=1e-3)[
        "exact"
    ]
    result = compare(mx.array([[1.0001]]), torch.tensor([[1.0]]), atol=1e-3, rtol=1e-3)
    assert not result["exact"] and result["max_abs_error"] > 0


def test_proposal_mismatch_is_not_accepted_as_a_near_tie():
    native = mx.array([[[18.25, 18.5]]])
    reference = torch.tensor([[[18.25, 18.25]]])
    # Floating-point tolerance alone would accept this difference.
    compare(native, reference, atol=0.25, rtol=0.02)
    with pytest.raises(AssertionError, match="row 0, position 0"):
        check_tokens(mx.array([[1]]), torch.tensor([[0]]), native, reference)


def test_proposal_shapes_must_match():
    with pytest.raises(AssertionError, match="shapes"):
        check_tokens(mx.array([[1]]), torch.tensor([[1, 2]]), None, None)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_reference_device_tensors_compare_and_report_mismatches(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    native = mx.array([[[1.0, 2.0]]])
    reference = torch.tensor([[[1.0, 2.0]]], device=device, requires_grad=True)
    tokens = torch.tensor([[1]], device=device)
    assert compare(native, reference, atol=0, rtol=0)["exact"]
    check_tokens(mx.array([[1]]), tokens, native, reference)
    with pytest.raises(AssertionError, match="native token 0, reference token 1"):
        check_tokens(mx.array([[0]]), tokens, native, reference)
