# SPDX-License-Identifier: Apache-2.0
"""The numerical qualification tool must fail on invalid or missing evidence."""

import mlx.core as mx
import pytest

from tools.dflash_parity import compare


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([[float("nan")]], [[float("nan")]]),
        ([[float("inf")]], [[float("inf")]]),
        ([], []),
        ([[1.0, 2.0]], [[1.0]]),
        ([[1.0]], [[1.1]]),
    ],
)
def test_comparison_rejects_nonfinite_incomplete_and_incorrect_results(
    actual, expected
):
    with pytest.raises((AssertionError, ValueError)):
        compare(mx.array(actual), mx.array(expected))


def test_comparison_distinguishes_numerical_tolerance_from_exact_identity():
    exact = compare(mx.array([[1.0]]), mx.array([[1.0]]))
    rounded = compare(mx.array([[1.0001]]), mx.array([[1.0]]))
    assert exact == {"exact": True, "max_abs_error": 0.0}
    assert rounded["exact"] is False and rounded["max_abs_error"] > 0
