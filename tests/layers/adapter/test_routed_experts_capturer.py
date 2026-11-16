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
"""TPU routed-experts capture: every forward of a step writes the table from
row 0, and its rows reach the step's snapshot at the step's row offset."""

from functools import partial
from types import SimpleNamespace
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo.utils
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector import worker as aux_output_worker
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata,
)
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu import utils
from vllm_torchtpu.distributed import aux_output_connector
from vllm_torchtpu.layers.adapter import routed_experts_capturer

NUM_LAYERS, TOPK = 3, 4
TOKEN_PADDINGS = [16, 32, 64]
MAX_NUM_BATCHED_TOKENS = 64
STAGING_ROWS = MAX_NUM_BATCHED_TOKENS + TOKEN_PADDINGS[-1]
BLOCK_SIZE = 4
# Marks padding rows; `_ids(..., num_experts=PADDING)` stays below it.
PADDING = 511
TPU = "tpu"


def _vllm_config(num_experts=512):
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


def _capturer(num_experts=512, with_staging=True):
    return routed_experts_capturer.TPURoutedExpertsCapturer(
        _vllm_config(num_experts), TOKEN_PADDINGS, with_staging=with_staging
    )


def _ids(num_tokens, num_experts=512, seed=0):
    """[num_tokens, layers, k] int32 expert ids, from the host generator only:
    seeding torch would also open the chip in this process."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(num_experts, size=(num_tokens, NUM_LAYERS, TOPK))
    return torch.from_numpy(ids.astype(np.int32))


def _forward(capturer, ids, device):
    """One forward's hook calls, under the forward context vLLM's hook reads."""
    with set_forward_context(None, _vllm_config(), num_tokens=ids.shape[0]):
        for layer in range(NUM_LAYERS):
            capturer.capture(layer, ids[:, layer].to(device))


@pytest.fixture
def cpu_capturer(monkeypatch):
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    return _capturer()


@pytest.mark.parametrize("is_warmup", [False, True])
def test_capture_traces_once_for_every_bucket(cpu_capturer, is_warmup):
    """One trace with a dynamic token dim, as vLLM compiles the backbone,
    writes every bucket's rows. A warmup forward's context holds 0 tokens."""
    torch._dynamo.reset()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    compiled = torch.compile(
        lambda ids: cpu_capturer.capture(1, ids),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )
    for num_tokens in (16, 32):
        ids = _ids(num_tokens)[:, 1]
        context_num_tokens = 0 if is_warmup else num_tokens
        with set_forward_context(None, _vllm_config(), num_tokens=context_num_tokens):
            compiled(ids)
        assert torch.equal(cpu_capturer.device_buffer[:num_tokens, 1], ids)

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs + 1
    assert not cpu_capturer.device_buffer[:, [0, 2]].any()


def test_capture_rejects_a_layer_past_the_table(cpu_capturer):
    with (
        set_forward_context(None, _vllm_config(), num_tokens=16),
        pytest.raises(IndexError, match="exceeds capture buffer"),
    ):
        cpu_capturer.capture(NUM_LAYERS, _ids(16)[:, 0])


@pytest.mark.parametrize("row_offset", [0, 7, STAGING_ROWS - 16])
def test_staging_writes_rows_at_the_offset(row_offset):
    staging = np.zeros((STAGING_ROWS, NUM_LAYERS, TOPK), dtype=np.uint16)
    rows = _ids(16).numpy()

    staged = routed_experts_capturer._stage_rows_jax(
        jnp.asarray(staging), jnp.asarray(rows), jnp.asarray([row_offset], jnp.int32)
    )

    expected = staging.copy()
    expected[row_offset : row_offset + 16] = rows
    assert np.array_equal(np.asarray(staged), expected)


def test_staging_past_the_table_raises(cpu_capturer):
    """dynamic_update_slice would clamp the start and overwrite other rows."""
    with pytest.raises(AssertionError, match="overruns"):
        cpu_capturer.stage_rows(16, STAGING_ROWS - 15)


def test_without_staging_only_the_table_is_written(monkeypatch):
    """Every rank can call stage_rows; only a staging rank has a snapshot."""
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    capturer = _capturer(with_staging=False)

    capturer.stage_rows(16, 0)

    with pytest.raises(AssertionError, match="with_staging"):
        capturer.snapshot_routing_data(16)


def test_snapshot_is_a_bucket_sized_copy(cpu_capturer):
    """Padded to the next bucket, and its own copy: the next step's staging
    writes must not reach a snapshot still being copied to the host."""
    first = cpu_capturer.snapshot_routing_data(20)
    second = cpu_capturer.snapshot_routing_data(20)
    first.fill_(1)

    assert first.shape[0] == 32 and first.dtype == torch.uint16
    assert not second.any()


def _stage_rows_on_host(staging, rows, row_offset):
    """The staging op's JAX body, for tests without a chip."""
    staged = routed_experts_capturer._stage_rows_jax(
        jnp.asarray(staging.numpy()),
        jnp.asarray(rows.numpy()),
        jnp.asarray(row_offset.numpy()),
    )
    return torch.from_numpy(np.array(staged))


class _MoELayer(torch.nn.Module):
    """A MoE layer as vLLM's capturer binder finds it."""

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn = None


def _moe_model():
    return torch.nn.ModuleList(_MoELayer(layer) for layer in range(NUM_LAYERS))


def _vllm_connector(monkeypatch, capturer, model=None):
    """vLLM's worker connector on the output rank, built through its
    constructor; on `capturer` unless that is None. Close it after use."""
    tp_group = SimpleNamespace(is_first_rank=True)
    monkeypatch.setattr(aux_output_worker, "get_tp_group", lambda: tp_group)
    model = _moe_model() if model is None else model
    connector = aux_output_worker.AuxOutputWorkerConnector(
        vllm_config=_vllm_config(),
        model=model,
        kv_cache_config=SimpleNamespace(num_blocks=1024, kv_cache_groups=[]),
    )
    if capturer is not None:
        aux_output_connector.use_capturer(connector, model, capturer)
    return connector


def _run_step(connector, capturer, ids, requests, block_hashes, finished=()):
    """One single-forward step of `ids` through the capturer and vLLM's
    connector; `requests` maps a request to (first token, tokens this step)."""
    connector.begin_step(
        AuxOutputConnectorMetadata(
            generation=0,
            requests=dict.fromkeys(requests, 0),
            block_hashes=block_hashes,
            finished_requests=finished,
        )
    )
    _forward(capturer, ids, "cpu")
    capturer.stage_rows(ids.shape[0], 0)
    token_starts, num_tokens = zip(*requests.values(), strict=True)
    pending = connector.prepare_output(
        SimpleNamespace(
            req_ids=list(requests),
            query_start_loc_np=np.cumsum((0, *num_tokens), dtype=np.int32),
            num_computed_tokens_np=np.array(token_starts, dtype=np.int32),
        )
    )
    pending.enqueue_cpu_copy(
        num_sampled=np.ones(len(requests), dtype=np.int32),
        num_rejected=np.zeros(len(requests), dtype=np.int32),
    )
    return pending.process_output()


@pytest.fixture
def cpu_connector(monkeypatch, cpu_capturer):
    connector = _vllm_connector(monkeypatch, cpu_capturer)
    yield connector
    connector.close()


def test_vllm_connector_never_reads_the_snapshot_padding(
    cpu_connector, cpu_capturer, monkeypatch
):
    """`snapshot_routing_data` returns a whole bucket where vLLM's returns the
    real row count, which is safe while vLLM's connector hands out and caches
    request rows only. A failure here means `snapshot_routing_data` has to
    change."""
    monkeypatch.setattr(routed_experts_capturer, "_STAGE_ROWS_OP", _stage_rows_on_host)
    step = partial(_run_step, cpu_connector, cpu_capturer)
    hashes = [b"a" * 32, b"b" * 32]
    # 10 real rows, then padding up to the bucket.
    ids = _ids(16, num_experts=PADDING)
    ids[10:] = PADDING

    outputs = step(ids, {"first": (0, 8), "second": (0, 2)}, {"first": hashes})

    assert (cpu_capturer.snapshot_routing_data(10)[10:] == PADDING).all()
    assert np.array_equal(outputs["first"].rows, ids[:8].numpy())
    assert np.array_equal(outputs["second"].rows, ids[8:10].numpy())

    # A prefix hit on both of `first`'s blocks reads them back from the cache.
    next_ids = _ids(16, num_experts=PADDING, seed=1)
    next_ids[1:] = PADDING
    outputs = step(next_ids, {"third": (8, 1)}, {"third": hashes}, ("first", "second"))

    expected = torch.cat([ids[:8], next_ids[:1]]).numpy()
    assert np.array_equal(outputs["third"].rows, expected)


# The tests below run on the chip, each in a spawned child, and
# spawns_tpu_workers runs them before tests that hold the chip in-process.


def _tpu_capturer(num_experts=512):
    from torch_tpu._internal import execution_mode

    # The engine runs eager ops in this mode.
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    capturer = _capturer(num_experts)
    # Otherwise the deferred zero fills join the first write's program.
    utils.synchronize_tensors()
    return capturer


def _step_rows_reach_the_snapshot_on_tpu(num_experts):
    capturer = _tpu_capturer(num_experts)
    # (real rows, bucket) of three forwards in one step.
    forwards = [(20, 32), (9, 16), (5, 16)]
    expected, row_offset = [], 0
    for seed, (num_tokens, bucket) in enumerate(forwards):
        ids = _ids(bucket, num_experts, seed)
        _forward(capturer, ids, TPU)
        capturer.stage_rows(bucket, row_offset)
        expected.append(ids[:num_tokens])
        row_offset += num_tokens
    host = capturer.snapshot_routing_data(row_offset).to("cpu", non_blocking=True)
    copied = torch.tpu.Event()
    copied.record()
    # The next step rewrites the staging table before that copy is waited on.
    _forward(capturer, _ids(16, num_experts, seed=9), TPU)
    capturer.stage_rows(16, 0)
    capturer.snapshot_routing_data(16).cpu()
    copied.synchronize()

    assert host.dtype == capturer.output_dtype
    assert host.shape[0] == 64
    assert np.array_equal(host[:row_offset].numpy(), torch.cat(expected).numpy())


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
@pytest.mark.parametrize("num_experts", [256, 512])
def test_step_rows_reach_the_snapshot_on_tpu(num_experts):
    """Each forward's real rows land after the previous forward's, in the
    connector's dtype (uint8 up to 256 experts, else uint16), and a snapshot
    keeps them while the next step rewrites the staging table."""
    run_in_isolated_process(partial(_step_rows_reach_the_snapshot_on_tpu, num_experts))


def _staging_compiles_once_per_bucket_on_tpu():
    capturer = _tpu_capturer()

    def stage(bucket, row_offset, num_tokens):
        capturer.stage_rows(bucket, row_offset)
        capturer.snapshot_routing_data(num_tokens).cpu()

    for bucket in TOKEN_PADDINGS:
        stage(bucket, 0, bucket)
    compiles = torch.tpu._get_cache_stats().num_compilations
    for bucket in TOKEN_PADDINGS:
        stage(bucket, 5, bucket - 1)
        stage(bucket, STAGING_ROWS - bucket, bucket // 2 + 1)

    assert torch.tpu._get_cache_stats().num_compilations == compiles


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_staging_compiles_once_per_bucket_on_tpu():
    """The offset is program data and the snapshot is a whole bucket, so a new
    offset or a new row count within a bucket compiles nothing.

    Each write is waited on, as the runner's forwards separate them: eager
    writes deferred back to back fuse into programs of their own.
    """
    run_in_isolated_process(_staging_compiles_once_per_bucket_on_tpu)
