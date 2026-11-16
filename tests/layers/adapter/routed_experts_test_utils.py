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
"""Shared fakes for the routed-experts capturer and connector tests."""

from types import SimpleNamespace
from typing import cast

import jax.numpy as jnp
import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector import worker as aux_output_worker
from vllm.forward_context import set_forward_context

from vllm_torchtpu.distributed import aux_output_connector
from vllm_torchtpu.layers.adapter import routed_experts_capturer

NUM_LAYERS, TOPK = 3, 4
TOKEN_PADDINGS = [16, 32, 64]
MAX_NUM_BATCHED_TOKENS = 64
BLOCK_SIZE = 4
# Marks padding rows; `expert_ids(..., num_experts=PADDING)` stays below it.
PADDING = 511


def make_vllm_config(num_experts=512):
    model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: NUM_LAYERS,
        get_num_experts=lambda: num_experts,
        get_num_experts_per_tok=lambda: TOPK,
        hf_text_config=SimpleNamespace(model_type="qwen3_moe"),
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, max_num_seqs=8
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            data_parallel_size=1,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=True,
        ),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE, prefix_match_unit=None),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
        aux_output_config=SimpleNamespace(max_bytes=None),
        max_concurrent_batches=2,
    )
    return cast(VllmConfig, vllm_config)


def make_capturer(num_experts=512, with_staging=True):
    return routed_experts_capturer.TPURoutedExpertsCapturer(
        make_vllm_config(num_experts), TOKEN_PADDINGS, with_staging=with_staging
    )


def expert_ids(num_tokens, num_experts=512, seed=0):
    """[num_tokens, layers, k] int32 expert ids, from the host generator only:
    seeding torch would also open the chip in this process."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(num_experts, size=(num_tokens, NUM_LAYERS, TOPK))
    return torch.from_numpy(ids.astype(np.int32))


def stage_rows_on_host(staging, rows, row_offset):
    """The staging op's JAX body, for tests without a chip."""
    staged = routed_experts_capturer._stage_rows_jax(
        jnp.asarray(staging.numpy()),
        jnp.asarray(rows.numpy()),
        jnp.asarray(row_offset.numpy()),
    )
    return torch.from_numpy(np.array(staged))


class MoELayer(torch.nn.Module):
    """A MoE layer as vLLM's capturer binder finds it."""

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn = None


def make_moe_model():
    return torch.nn.ModuleList(MoELayer(layer) for layer in range(NUM_LAYERS))


def model_forward(model, ids):
    """One forward of [bucket, layers, k] `ids` through the hooks on `model`."""
    with set_forward_context(None, make_vllm_config(), num_tokens=ids.shape[0]):
        for layer in model:
            layer.capture_fn(ids[:, layer.layer_id])


def make_vllm_connector(monkeypatch, capturer, model=None):
    """vLLM's worker connector on the output rank, built through its
    constructor; on `capturer` unless that is None. Close it after use."""
    tp_group = SimpleNamespace(is_first_rank=True)
    monkeypatch.setattr(aux_output_worker, "get_tp_group", lambda: tp_group)
    model = make_moe_model() if model is None else model
    connector = aux_output_worker.AuxOutputWorkerConnector(
        vllm_config=make_vllm_config(),
        model=model,
        kv_cache_config=SimpleNamespace(num_blocks=1024, kv_cache_groups=[]),
    )
    if capturer is not None:
        aux_output_connector.use_capturer(connector, model, capturer)
    return connector
