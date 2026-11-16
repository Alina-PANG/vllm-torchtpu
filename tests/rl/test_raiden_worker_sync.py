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
"""CPU unit tests for RaidenWorkerSync parameter classification and wire metadata."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.rl import raiden_worker_sync


class _FakeRMSNorm(GemmaRMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )


class _FakeVocabEmbedding(VocabParallelEmbedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_size: int,
        *,
        org_vocab_size: int | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.org_vocab_size = org_vocab_size or vocab_size
        self.weight = nn.Parameter(
            torch.zeros(vocab_size // tp_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _FakeLMHead(ParallelLMHead):
    def __init__(self, vocab_size: int, hidden_size: int, tp_size: int) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.org_vocab_size = vocab_size
        self.weight = nn.Parameter(
            torch.zeros(vocab_size // tp_size, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _FakeReplicatedLinear(ReplicatedLinear):
    def __init__(
        self, in_features: int, out_features: int, *, flipped: bool = False
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = 1
        shape = (in_features, out_features) if flipped else (out_features, in_features)
        self.weight = nn.Parameter(
            torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False
        )
        if flipped:
            setattr(self, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)


class _FakeRowLinear(RowParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        *,
        flipped: bool = True,
        bias: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        local_in = in_features // tp_size
        shape = (local_in, out_features) if flipped else (out_features, local_in)
        self.weight = nn.Parameter(
            torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16), requires_grad=False
            )
        if flipped:
            setattr(self, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)


class _FakeQKVLinear(QKVParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_q_heads: int,
        total_kv_heads: int,
        tp_size: int,
        *,
        flipped: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.head_size = head_size
        self.num_heads = total_q_heads // tp_size
        self.total_num_kv_heads = total_kv_heads
        self.num_kv_head_replicas = max(1, tp_size // total_kv_heads)
        local_kv = max(1, total_kv_heads // tp_size)
        local_out = (self.num_heads + 2 * local_kv) * head_size
        shape = (hidden_size, local_out) if flipped else (local_out, hidden_size)
        self.weight = nn.Parameter(
            torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False
        )
        if flipped:
            setattr(self, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)


class _FakeGdnColumnLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        checkpoint_sizes: list[int],
        replication_factors: list[int],
        tp_size: int,
        *,
        is_conv: bool = False,
        flipped: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.checkpoint_sizes = list(checkpoint_sizes)
        self.replication_factors = list(replication_factors)
        self.output_sizes = [
            s * r for s, r in zip(checkpoint_sizes, replication_factors)
        ]
        local_out = sum(s // tp_size for s in self.output_sizes)
        if is_conv:
            self.weight = nn.Parameter(
                torch.zeros((local_out, 1, in_features), dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            shape = (in_features, local_out) if flipped else (local_out, in_features)
            self.weight = nn.Parameter(
                torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False
            )
            if flipped:
                setattr(self, raiden_worker_sync.WEIGHT_FLIPPED_ATTR, True)


class _FakeRoutedExperts(RoutedExperts):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        inter_size: int,
        *,
        use_ep: bool,
        ep_size: int,
        moe_tp_size: int,
    ) -> None:
        nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=use_ep,
                ep_size=ep_size if use_ep else 1,
                tp_size=1 if use_ep else moe_tp_size,
            )
        )
        self._tpu_activation = "silu"
        local_e = num_experts // ep_size if use_ep else num_experts
        half = inter_size if use_ep else (inter_size // moe_tp_size)
        aligned_half = (half + 127) // 128 * 128
        self.w13_weight = nn.Parameter(
            torch.zeros((local_e, hidden_size, 2 * aligned_half), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.zeros((local_e, aligned_half, hidden_size), dtype=torch.bfloat16),
            requires_grad=False,
        )


class _FakeMambaLayer(MambaBase, nn.Module):
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32, torch.float32)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((1,), (1,))

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def __init__(self, tp_size: int, num_v_heads: int = 8) -> None:
        nn.Module.__init__(self)
        self.tp_size = tp_size
        self.A_log = nn.Parameter(
            torch.zeros(num_v_heads // tp_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.ones(num_v_heads // tp_size, dtype=torch.bfloat16),
            requires_grad=False,
        )


def _assert_raiden_metadata_invariants(meta: dict) -> None:
    mesh_shape = meta["mesh_shape"]
    mesh_axes = meta["mesh_axes"]
    assert len(mesh_shape) == len(mesh_axes)
    assert len(set(mesh_axes)) == len(mesh_axes)
    phys = dict(zip(mesh_axes, mesh_shape))
    assert meta["unit"]["job_replica_id"] != ""
    assert len(meta["shards"]) >= 1

    seen: set[tuple[str, int]] = set()
    for idx, v in enumerate(meta["variables"]):
        assert v["layer_idx"] == idx
        key = (v["name"], v["layer_idx"])
        assert key not in seen
        seen.add(key)
        assert len(v["global_shard_indices"]) == len(meta["shards"])
        for dim, axis in enumerate(v["sharding_spec"]):
            logical = v["mesh_shape"][dim]
            if not axis:
                assert logical == 1
            else:
                parts = axis.split(",")
                for p in parts:
                    assert p in phys
                assert logical == math.prod(phys[p] for p in parts)


def test_canonical_param_name_and_post_strip_sort_order() -> None:
    assert (
        raiden_worker_sync.canonical_param_name(
            "language_model.model.embed_tokens.weight"
        )
        == "embed_tokens.weight"
    )
    assert (
        raiden_worker_sync.canonical_param_name(
            "model.layers.3.mlp.experts.routed_experts.w13_weight"
        )
        == "layers.3.mlp.experts.w13_weight"
    )
    assert (
        raiden_worker_sync.canonical_param_name(
            "base.model.layers_10.mlp.w13_weight_scale_2.value"
        )
        == "layers.10.mlp.w13_weight_scale.2"
    )

    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = _FakeVocabEmbedding(32, 8, tp_size=2)
    lm.model.layers = nn.ModuleDict(
        {
            "2": _FakeRMSNorm(8),
            "10": _FakeRMSNorm(8),
        }
    )
    lm.lm_head = _FakeLMHead(32, 8, tp_size=2)

    vars_sorted = raiden_worker_sync.collect_variables(lm, tp_size=2)
    assert [v.name for v in vars_sorted] == [
        "embed_tokens.weight",
        "layers.10.weight",
        "layers.2.weight",
        "lm_head.weight",
    ]


def test_vlm_tower_and_mtp_skipping_plus_tied_lm_head() -> None:
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = _FakeVocabEmbedding(32, 8, tp_size=2)
    lm.lm_head = _FakeLMHead(32, 8, tp_size=2)
    lm.lm_head.weight = lm.model.embed_tokens.weight
    lm.mtp = nn.Linear(8, 8)

    vlm = nn.Module()
    vlm.language_model = lm
    vision = nn.Linear(8, 8)
    vision._is_tower_model = True
    vlm.vision_tower = vision
    vlm.get_language_model = lambda: lm

    vars_bound = raiden_worker_sync.collect_variables(vlm, tp_size=2)
    assert [v.name for v in vars_bound] == ["embed_tokens.weight"]

    padded = nn.Module()
    padded.embed_tokens = _FakeVocabEmbedding(32, 8, tp_size=2, org_vocab_size=30)
    with pytest.raises(ValueError, match="padded vocab"):
        raiden_worker_sync.collect_variables(padded, tp_size=2)

    dp_vocab = nn.Module()
    dp_vocab.embed_tokens = _FakeVocabEmbedding(64, 8, tp_size=8)
    dp_vocab.embed_tokens._tpu_dp_vocab = True
    (dp_var,) = raiden_worker_sync.collect_variables(dp_vocab, tp_size=4)
    assert dp_var.shard_domain == "ep"
    assert dp_var.num_shards == 8


def test_owning_module_isinstance_classification() -> None:
    root = nn.Module()
    root.rep_unflipped = _FakeReplicatedLinear(16, 4, flipped=False)
    root.rep_flipped = _FakeReplicatedLinear(16, 4, flipped=True)
    root.row_flipped = _FakeRowLinear(16, 8, tp_size=4, flipped=True, bias=True)
    root.row_unflipped = _FakeRowLinear(16, 8, tp_size=4, flipped=False)
    root.qkv = _FakeQKVLinear(
        16, head_size=4, total_q_heads=8, total_kv_heads=2, tp_size=4
    )
    root.gdn_qkvz_unflipped = _FakeGdnColumnLinear(
        16, [8, 8, 16, 16], [2, 2, 1, 1], tp_size=4, flipped=False
    )
    root.gdn_qkvz_flipped = _FakeGdnColumnLinear(
        16, [8, 8, 16, 16], [2, 2, 1, 1], tp_size=4, flipped=True
    )
    root.conv = _FakeGdnColumnLinear(4, [8, 8, 16], [2, 2, 1], tp_size=4, is_conv=True)
    root.mamba = _FakeMambaLayer(tp_size=4, num_v_heads=8)
    root.moe_tp_block = _FakeRoutedExperts(
        4, 256, 768, use_ep=False, ep_size=1, moe_tp_size=4
    )

    by_name = {v.name: v for v in raiden_worker_sync.collect_variables(root, tp_size=4)}

    assert by_name["rep_unflipped.weight"].shard_dim is None
    assert by_name["rep_unflipped.weight"].flipped is False
    assert by_name["rep_flipped.weight"].shard_dim is None
    assert by_name["rep_flipped.weight"].flipped is True

    assert by_name["row_flipped.weight"].shard_dim == 0
    assert by_name["row_flipped.bias"].shard_dim is None
    assert by_name["row_unflipped.weight"].shard_dim == 1

    qkv = by_name["qkv.weight"]
    assert qkv.shard_dim == 1
    assert qkv.num_shards == 4
    assert qkv.flipped is True

    assert by_name["gdn_qkvz_unflipped.weight"].shard_dim == 0
    assert by_name["gdn_qkvz_unflipped.weight"].flipped is False
    assert by_name["gdn_qkvz_flipped.weight"].shard_dim == 1
    assert by_name["gdn_qkvz_flipped.weight"].flipped is True

    assert by_name["conv.weight"].shard_dim == 0
    assert "conv_transposed" not in by_name["conv.weight"].layout_dict()

    assert by_name["mamba.A_log"].shard_dim == 0
    assert by_name["mamba.dt_bias"].shard_dim == 0

    # 768 // 4 = 192 -> padded to 256 per half -> local w13_weight is [4, 256, 512]
    assert by_name["moe_tp_block.w13_weight"].local_shape == (4, 256, 512)
    assert by_name["moe_tp_block.w13_weight"].global_shape == (4, 256, 2048)
    assert by_name["moe_tp_block.w13_weight"].shard_dim == 2
    assert by_name["moe_tp_block.w2_weight"].local_shape == (4, 256, 256)
    assert by_name["moe_tp_block.w2_weight"].global_shape == (4, 1024, 256)
    assert by_name["moe_tp_block.w2_weight"].shard_dim == 1


def test_dp_tp_ep_metadata_checksums_and_bind_lifecycle() -> None:
    model = nn.Module()
    model.embed_tokens = _FakeVocabEmbedding(64, 16, tp_size=4)
    model.experts = _FakeRoutedExperts(
        16, 16, 128, use_ep=True, ep_size=8, moe_tp_size=1
    )
    model.norm = _FakeRMSNorm(16)
    model.norm.weight.data.fill_(2.0)

    fake_ws_cls = MagicMock()
    fake_ws_inst = MagicMock()
    fake_ws_inst.num_shards = 1
    fake_ws_inst.local_port = 19500
    fake_ws_inst.listener_port = 29500
    fake_ws_inst.get_local_endpoints.return_value = [
        {"endpoint": "[::]:19500", "shards": [0]}
    ]
    fake_ws_inst.get_metrics.return_value = {"bytes": 1024}
    fake_ws_cls.return_value = fake_ws_inst

    fake_ws_mod = SimpleNamespace(WeightSynchronizer=fake_ws_cls)

    with (
        patch.object(raiden_worker_sync, "_ws_lib", fake_ws_mod),
        patch.object(raiden_worker_sync, "_check_bindable"),
        patch("torch.accelerator.synchronize"),
    ):
        ws_00 = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            worker_index=0,
            tp_rank=0,
            tp_size=4,
            dp_rank=0,
            dp_size=2,
            bind_ip="[2001:db8::1]",
            auto_h2d=False,
        )
        assert not ws_00.bound
        assert ws_00.checksums() == {
            "__grand_total__": 0.0,
            "__tensor_count__": 0,
            "__element_count__": 0,
        }
        ws_00.bind(model)
        assert ws_00.bound
        # 1-chip worker must always pass global_shard_indices=[0] to C++ constructor.
        assert fake_ws_cls.call_args.kwargs["global_shard_indices"] == [0]
        meta_00 = ws_00.metadata_dict()
        _assert_raiden_metadata_invariants(meta_00)
        assert meta_00["shards"] == ["[2001:db8::1]:19500"]
        assert meta_00["unit"]["job_replica_id"] == "0"
        assert "param_layouts" in meta_00
        assert "embed_tokens.weight" in ws_00.describe()

        csums = ws_00.checksums(sample=2)
        assert csums["__tensor_count__"] == 4
        assert csums["__element_count__"] == (
            16 * 16 + 2 * 16 * 256 + 2 * 128 * 16 + 16
        )
        assert csums["__grand_total__"] == pytest.approx(32.0)

        # Rebinding with mismatched auto_h2d preserves original setting and warns.
        ws_00.bind(model, auto_h2d=True)
        assert ws_00.metadata_dict()["auto_h2d"] is False

        ws_13 = raiden_worker_sync.RaidenWorkerSync(
            "replica_0",
            worker_index=0,
            tp_rank=3,
            tp_size=4,
            dp_rank=1,
            dp_size=2,
            bind_ip="10.0.0.1",
            auto_h2d=False,
        )
        ws_13.bind(model)
        meta_13 = ws_13.metadata_dict()
        _assert_raiden_metadata_invariants(meta_13)
        assert meta_13["unit"]["job_replica_id"] == "7"
        assert "param_layouts" not in meta_13

        ep_var = next(
            v for v in meta_13["variables"] if v["name"] == "experts.w13_weight"
        )
        assert ep_var["sharding_spec"] == ["dp,tp", "", ""]
        assert ep_var["mesh_shape"] == [8, 1, 1]
        assert ep_var["global_shard_indices"] == [7]

        # Test h2d validation under auto_h2d=False and re-binding when a pointer moves.
        with pytest.raises(ValueError, match="uuid > 0"):
            ws_00.h2d(uuid=0)
        ws_00.h2d(uuid=3)
        fake_ws_inst.wait_for_transfer_completion.assert_called_once_with(3)
        fake_ws_inst.h2d.assert_called_once()

        model.norm.weight = nn.Parameter(torch.ones_like(model.norm.weight))
        ws_00.bind(model)
        fake_ws_inst.bind_weights.assert_called_once()

        # Sub-mesh MoE TP mismatch against physical worker tp_size fails fast.
        bad_moe_model = nn.Module()
        bad_moe_model.experts = _FakeRoutedExperts(
            4, 16, 128, use_ep=False, ep_size=1, moe_tp_size=2
        )
        ws_bad = raiden_worker_sync.RaidenWorkerSync(
            "replica_bad", tp_rank=0, tp_size=4, bind_ip="10.0.0.1"
        )
        ws_bad.bind(bad_moe_model)
        with pytest.raises(NotImplementedError, match="TP shard count 2"):
            ws_bad.variable_dicts()
