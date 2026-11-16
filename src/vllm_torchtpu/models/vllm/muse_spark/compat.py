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
"""Idempotent TPU compatibility helpers for upstream ``vllm.models.muse_spark``."""

from __future__ import annotations

import torch
from torch import nn

_APPLIED = False


def apply() -> None:
    """Gate upstream ``MuseSpark1xRMSNorm.supports_fused_kernel`` on CUDA.

    Upstream ``vllm.models.muse_spark.common.layers.MuseSpark1xRMSNorm`` dispatches
    to its Triton ``fused_rmsnorm_fwd`` kernel whenever ``x`` is contiguous and
    matches ``hidden_size``, without checking ``x.is_cuda``. Adding ``x.is_cuda``
    routes TPU and CPU tensors cleanly through ``forward_native``.
    """
    global _APPLIED
    if _APPLIED:
        return

    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm

    orig_supports = MuseSpark1xRMSNorm.supports_fused_kernel

    def _supports_fused_kernel_cuda_only(
        self: MuseSpark1xRMSNorm, x: torch.Tensor
    ) -> bool:
        return bool(x.is_cuda and orig_supports(self, x))

    MuseSpark1xRMSNorm.supports_fused_kernel = _supports_fused_kernel_cuda_only  # type: ignore[method-assign]
    _APPLIED = True


def sync_residual_constants(model: nn.Module) -> None:
    """Synchronize newly registered residual-gate ``alpha``/``beta`` buffers on TPU."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    from vllm_torchtpu.utils import synchronize_tensors

    tpu_buffers: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, MuseSpark1xNormPreservingResidualConnection):
            for name in ("alpha", "beta"):
                buf = getattr(module, name, None)
                if (
                    buf is not None
                    and isinstance(buf, torch.Tensor)
                    and buf.device.type == "tpu"
                ):
                    tpu_buffers.append(buf)
    if tpu_buffers:
        synchronize_tensors(tpu_buffers)
