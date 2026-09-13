# SPDX-License-Identifier: Apache-2.0
"""Granite hybrid topology and Mamba-2 state geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper, is_mamba2_mixer
from vllm_metal.attention.runtime.families.gdn import create_gdn_state_cache
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridLayerPlan,
    HybridRuntimePlan,
    RecurrentStateGeometry,
    StateFamilySpec,
)

GRANITE_MODEL_TYPES = frozenset({"granitemoehybrid"})

GRANITE_FAMILY = StateFamilySpec(
    label="granite",
    wrapper_cls=Mamba2PagedStateWrapper,
    is_state_module=is_mamba2_mixer,
    mamba_type=MambaAttentionBackendEnum.MAMBA2,
    supported_cache_modes=("none",),
    layer_name="mamba",
    create_state_cache=create_gdn_state_cache,
)


def build_granite_hybrid_plan(
    model_args: Mapping[str, Any],
    num_layers: int,
    state_dtypes: tuple[torch.dtype, ...],
) -> HybridRuntimePlan:
    """Resolve Granite's explicit layer types and mlx-lm Mamba-2 dimensions."""
    dimension_names = (
        "mamba_n_heads",
        "mamba_d_head",
        "mamba_d_state",
        "mamba_n_groups",
        "mamba_d_conv",
    )
    try:
        layer_types = tuple(model_args["layer_types"])
        dims = {name: model_args[name] for name in dimension_names}
    except KeyError as exc:
        raise ValueError(
            f"Granite hybrid model args are missing required {exc.args[0]!r}."
        ) from exc

    if len(layer_types) != num_layers:
        raise ValueError(
            f"Granite hybrid layer_types has {len(layer_types)} entries, "
            f"but num_layers={num_layers}."
        )
    if set(layer_types) != {"mamba", "attention"}:
        raise ValueError(
            "Granite hybrid layer_types must contain both 'mamba' and "
            f"'attention' and no other layer types, got {layer_types!r}."
        )
    invalid = [
        f"{name}={value!r}"
        for name, value in dims.items()
        if type(value) is not int or value <= 0
    ]
    if invalid:
        raise ValueError(
            "Granite hybrid state dimensions must be positive integers; "
            f"invalid {', '.join(invalid)}."
        )
    if dims["mamba_n_heads"] % dims["mamba_n_groups"]:
        raise ValueError(
            "Granite hybrid mamba_n_heads must be divisible by mamba_n_groups."
        )

    return HybridRuntimePlan(
        layers=HybridLayerPlan(
            layer_roles=tuple(
                STATE_LAYER if kind == "mamba" else ATTENTION_LAYER
                for kind in layer_types
            )
        ),
        family=GRANITE_FAMILY,
        geometry=RecurrentStateGeometry(
            conv_kernel_dim=dims["mamba_d_conv"],
            conv_dim=dims["mamba_n_heads"] * dims["mamba_d_head"]
            + 2 * dims["mamba_n_groups"] * dims["mamba_d_state"],
            num_v_heads=dims["mamba_n_heads"],
            value_head_dim=dims["mamba_d_head"],
            key_head_dim=dims["mamba_d_state"],
        ),
        state_dtypes=state_dtypes,
    )
