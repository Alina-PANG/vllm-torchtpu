# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Out-of-tree TPU custom operation and layer registration for MambaMixer3."""

from __future__ import annotations

import functools
from typing import Any, cast

import jax
import torch
from vllm.config import VllmConfig, get_current_vllm_config_or_none

from vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils import (
    extract_mamba_step_metadata,
    get_shared_mamba_jax_op,
    unpack_mamba_kv_cache,
    validate_tpu_mamba_config,
)
from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3
from vllm_torchtpu.layers.mamba_mixer3 import MambaMixer3
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)


def mamba_mixer3_core_tpu(
    projected_states: jax.Array,
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    B_norm_weight: jax.Array,
    C_norm_weight: jax.Array,
    B_bias: jax.Array,
    C_bias: jax.Array,
    D: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array | None,
    psi: jax.Array | None,
    phi: jax.Array | None,
    zeta: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int,
    is_mimo: bool,
    mimo_rank: int,
    A_floor: float,
    rms_norm_eps: float,
    apply_rms_norm: bool,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Slice padded ``aux_state`` trailing width if needed and run Mamba-3 JAX core."""
    r = mimo_rank if is_mimo else 1
    target_aux_dim = r * (head_dim + d_state) + num_rope_angles
    aux_dim = aux_state.shape[-1]
    aux_state_in = (
        aux_state[..., :target_aux_dim] if aux_dim > target_aux_dim else aux_state
    )

    new_aux_state, new_recurrent_state, output = run_jax_mamba_mixer3(
        projected_states,
        aux_state_in,
        recurrent_state,
        B_norm_weight,
        C_norm_weight,
        B_bias,
        C_bias,
        D,
        dt_bias,
        norm_weight,
        psi,
        phi,
        zeta,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        mesh=mesh,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        num_rope_angles=num_rope_angles,
        chunk_size=chunk_size,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        A_floor=A_floor,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
        interpret=interpret,
    )
    if aux_dim > target_aux_dim:
        new_aux_state = aux_state.at[..., :target_aux_dim].set(new_aux_state)
    return new_aux_state, new_recurrent_state, output


@MambaMixer3.register_oot
class VllmMambaMixer3(MambaMixer3):
    """Out-of-tree TPU implementation of MambaMixer3 backed by JAX/Pallas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._apply_py_norm = False
        self.mamba_op = self._build_mamba_op()

    def _use_fused_jax_rms_norm(self) -> bool:
        return self.norm.weight is None or self.norm.n_groups % self.tp_size == 0

    def _build_mamba_op(self, *, interpret: bool = False):
        try:
            vllm_context = get_vllm_model_wrapper_context()
        except AssertionError:
            vllm_context = None
        resolved_vllm_config = (
            cast(VllmConfig | None, vllm_context.vllm_config)
            if vllm_context is not None and vllm_context.vllm_config is not None
            else get_current_vllm_config_or_none()
        )
        validate_tpu_mamba_config(
            "MambaMixer3",
            cache_config=self.cache_config,
            vllm_config=resolved_vllm_config,
        )
        if vllm_context is None:
            vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        apply_rms_norm = self._use_fused_jax_rms_norm()
        self._apply_py_norm = not apply_rms_norm
        rms_norm_eps = self.norm.variance_epsilon
        out_dim = self.tped_dt_size * self.head_dim

        op_key = (
            id(mesh),
            self.tped_dt_size,
            self.head_dim,
            self.tp_n_groups,
            self.ssm_state_size,
            self.num_rope_angles,
            self.chunk_size,
            self.is_mimo,
            self.mimo_rank,
            self.A_floor,
            rms_norm_eps,
            apply_rms_norm,
            interpret,
        )
        wrapped_fn = functools.partial(
            mamba_mixer3_core_tpu,
            mesh=mesh,
            num_heads=self.tped_dt_size,
            head_dim=self.head_dim,
            n_groups=self.tp_n_groups,
            d_state=self.ssm_state_size,
            num_rope_angles=self.num_rope_angles,
            chunk_size=self.chunk_size,
            is_mimo=self.is_mimo,
            mimo_rank=self.mimo_rank,
            A_floor=self.A_floor,
            rms_norm_eps=rms_norm_eps,
            apply_rms_norm=apply_rms_norm,
            interpret=interpret,
        )
        return get_shared_mamba_jax_op("mamba_mixer3", op_key, wrapped_fn, out_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        out_dim = self.tped_dt_size * self.head_dim

        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        unpacked_cache = unpack_mamba_kv_cache(self.kv_cache, "MambaMixer3")
        if unpacked_cache is None:
            ssm_output = torch.zeros(
                (num_tokens, out_dim),
                dtype=projected_states.dtype,
                device=projected_states.device,
            )
            apply_py_norm = True
        else:
            aux_state, recurrent_state = unpacked_cache
            state_indices, query_start_loc, request_distribution, seq_lens = (
                extract_mamba_step_metadata(self.prefix)
            )
            apply_py_norm = self._apply_py_norm
            norm_weight = cast(torch.Tensor | None, self.norm.weight)
            ssm_output = self.mamba_op(
                projected_states,
                aux_state,
                recurrent_state,
                self.B_norm_weight,
                self.C_norm_weight,
                self.B_bias,
                self.C_bias,
                self.D,
                self.dt_bias,
                norm_weight,
                self.Psi,
                self.Phi,
                self.Zeta,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
            )

        if apply_py_norm:
            gate = (
                None
                if self.is_mimo
                else projected_states[..., : self.tped_intermediate_size]
            )
            hidden_states = self.norm(ssm_output, gate)
        else:
            hidden_states = ssm_output

        output, _ = self.out_proj(hidden_states)
        return output
