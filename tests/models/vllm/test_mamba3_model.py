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
"""Unit and TPU tests for Mamba3ForCausalLM, Mamba3Model, and Mamba3Config."""

from __future__ import annotations

import json
import multiprocessing
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, PretrainedConfig
from vllm.config import (
    CacheConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.interfaces import SupportsMambaPrefixCaching
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import get_config
from vllm.v1.kv_cache_interface import MambaSpec

from tests.layers.adapter.mamba_test_utils import (
    eval_hf_mamba2_reference,
    eval_torch_mamba3_reference,
    jax_cpu_mesh,
    make_forward_context,
    make_mamba_attn_metadata,
    passthrough_jax_op,
    write_synthetic_mamba2_checkpoint,
    write_synthetic_mamba3_checkpoint,
)
from vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op import VllmMambaMixer3
from vllm_torchtpu.layers.mamba_mixer3 import MambaMixer3
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.mamba3.config import (
    Mamba3Config,
    TPUMamba3Config,
)
from vllm_torchtpu.models.vllm.mamba3.mamba3 import (
    Mamba3ForCausalLM,
    Mamba3GatedMLP,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture(autouse=True)
def _cpu_platform_for_unit_tests(request, monkeypatch):
    """Pin `current_platform` to CPU before `dist_ctx` with clean teardown."""
    if request.node.get_closest_marker("tpu_test") is not None:
        return
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)
    request.getfixturevalue("dist_ctx")


def _make_vllm_config(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> VllmConfig:
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        dtype=dtype,
        head_dtype=None,
        max_model_len=256,
    )
    vllm_config.cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_block_size=256,
        mamba_page_size_padded=None,
    )
    vllm_config.speculative_config = None
    vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        pipeline_parallel_size=pp_size,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    vllm_config.compilation_config.mode = CompilationMode.NONE
    return vllm_config


def _build_mamba3_model(
    hf_config: Mamba3Config,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_group: Any = None,
    pp_layer_range: tuple[int, int] | None = None,
    use_real_jax_op: bool = False,
) -> tuple[Mamba3ForCausalLM, VllmConfig]:
    pp_size = 2 if pp_group is not None else 1
    vllm_config = _make_vllm_config(hf_config, tp_size=tp_size, pp_size=pp_size)
    mesh = (
        jax_cpu_mesh()
        if use_real_jax_op
        else SimpleNamespace(shape={"model": 1}, axis_names=("model",))
    )

    with ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(vllm_config))
        stack.enter_context(
            set_vllm_model_wrapper_context(mesh=mesh, vllm_config=vllm_config)
        )
        if pp_group is not None:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
                    return_value=pp_group,
                )
            )
        if pp_layer_range is not None:
            start_l, end_l = pp_layer_range

            def _fake_make_layers(num_hidden_layers: int, layer_fn: Any, prefix: str):
                layers = torch.nn.ModuleList(
                    [
                        layer_fn(prefix=f"{prefix}.{idx}")
                        if start_l <= idx < end_l
                        else PPMissingLayer()
                        for idx in range(num_hidden_layers)
                    ]
                )
                return start_l, end_l, layers

            stack.enter_context(
                patch(
                    "vllm_torchtpu.models.vllm.mamba3.mamba3.make_layers",
                    side_effect=_fake_make_layers,
                )
            )
        if tp_size > 1:
            for target in (
                "vllm_torchtpu.layers.mamba_mixer3.get_tensor_model_parallel_world_size",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
                "vllm.model_executor.layers.mamba.mamba_mixer2.get_tensor_model_parallel_world_size",
                "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            ):
                stack.enter_context(patch(target, return_value=tp_size))
            for target in (
                "vllm_torchtpu.layers.mamba_mixer3.get_tensor_model_parallel_rank",
                "vllm_torchtpu.models.vllm.mamba3.mamba3.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
                "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            ):
                stack.enter_context(patch(target, return_value=tp_rank))
        if use_real_jax_op:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                    side_effect=passthrough_jax_op,
                )
            )
        else:
            stack.enter_context(
                patch(
                    "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                    return_value=MagicMock(),
                )
            )

        model = Mamba3ForCausalLM(vllm_config=vllm_config)
        if use_real_jax_op:
            for decoder_layer in model.backbone.layers:
                if not isinstance(decoder_layer, PPMissingLayer):
                    decoder_layer.mixer.mamba_op = decoder_layer.mixer._build_mamba_op(
                        interpret=True
                    )

    return model, vllm_config


class TestMamba3RegistrationAndConfig:
    def test_mamba3_model_and_config_map_registration(self):
        register_models()
        entry = ModelRegistry.models.get("Mamba3ForCausalLM")
        assert entry is not None, "Mamba3ForCausalLM is not registered"
        assert entry.load_model_cls() is Mamba3ForCausalLM
        assert MODELS_CONFIG_MAP.get("Mamba3ForCausalLM") is TPUMamba3Config

    def test_scoped_vllm_config_parser_without_global_monkeypatch(self, tmp_path):
        register_models()

        siso_dir = tmp_path / "mamba3-siso-1.5b"
        siso_dir.mkdir()
        siso_json = {
            "d_model": 2048,
            "d_intermediate": 4096,
            "n_layer": 24,
            "vocab_size": 128256,
            "ssm_cfg": {
                "layer": "Mamba3",
                "d_state": 128,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "A_floor": 0.0001,
                "chunk_size": 64,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "tie_embeddings": True,
            "pad_vocab_size_multiple": 16,
        }
        (siso_dir / "config.json").write_text(json.dumps(siso_json))

        # Raw HF PretrainedConfig.get_config_dict stays untouched outside vLLM.
        raw_dict, _ = PretrainedConfig.get_config_dict(str(siso_dir))
        assert "model_type" not in raw_dict
        assert "architectures" not in raw_dict

        # vLLM's scoped "hf" config parser resolves ssm_cfg.layer == "Mamba3".
        siso_cfg = get_config(str(siso_dir), trust_remote_code=False)
        assert isinstance(siso_cfg, Mamba3Config)
        assert siso_cfg.model_type == "mamba3"
        assert siso_cfg.architectures == ["Mamba3ForCausalLM"]
        assert siso_cfg.hidden_size == 2048
        assert siso_cfg.num_hidden_layers == 24
        assert siso_cfg.d_intermediate == 4096
        assert siso_cfg.num_heads == 64
        assert siso_cfg.head_dim == 64
        assert siso_cfg.state_size == 128
        assert siso_cfg.is_mimo is False
        assert siso_cfg.mimo_rank == 1

        # Explicit model_type="mamba3" also works directly via AutoConfig.
        explicit_dir = tmp_path / "mamba3-explicit"
        explicit_dir.mkdir()
        (explicit_dir / "config.json").write_text(
            json.dumps(dict(siso_json, model_type="mamba3"))
        )
        assert isinstance(AutoConfig.from_pretrained(str(explicit_dir)), Mamba3Config)

        # Verify to_dict() -> from_dict() roundtrip preserves normalized fields
        # including unaligned org_vocab_size vs padded vocab_size.
        unaligned = Mamba3Config(vocab_size=100, pad_vocab_size_multiple=16)
        assert unaligned.org_vocab_size == 100
        assert unaligned.vocab_size == 112
        roundtrip = Mamba3Config.from_dict(unaligned.to_dict())
        assert roundtrip.org_vocab_size == 100
        assert roundtrip.vocab_size == 112

        mimo_dir = tmp_path / "mamba3-mimo-1.5b"
        mimo_dir.mkdir()
        mimo_json = dict(siso_json, d_intermediate=3824)
        mimo_json["ssm_cfg"] = dict(
            siso_json["ssm_cfg"], is_mimo=True, mimo_rank=4, chunk_size=16
        )
        (mimo_dir / "config.json").write_text(json.dumps(mimo_json))
        mimo_cfg = get_config(str(mimo_dir), trust_remote_code=False)
        assert isinstance(mimo_cfg, Mamba3Config)
        assert mimo_cfg.is_mimo is True
        assert mimo_cfg.mimo_rank == 4
        assert mimo_cfg.chunk_size == 16
        assert mimo_cfg.d_intermediate == 3824

    def test_engine_startup_vllm_config_initializes_mamba_block_size(self, tmp_path):
        """Verifies `MODELS_CONFIG_MAP["Mamba3ForCausalLM"]` sets `mamba_block_size`
        so `MambaBase.get_kv_cache_spec` succeeds on startup without AssertionError."""
        register_models()
        ckpt_dir = tmp_path / "mamba3_startup"
        write_synthetic_mamba3_checkpoint(ckpt_dir, is_mimo=False)

        model_cfg = ModelConfig(
            model=str(ckpt_dir),
            tokenizer=str(ckpt_dir),
            max_model_len=128,
            dtype="bfloat16",
        )
        vllm_cfg = VllmConfig(
            model_config=model_cfg,
            cache_config=CacheConfig(enable_prefix_caching=False),
            device_config=DeviceConfig(device="cpu"),
        )
        vllm_cfg.compilation_config.mode = CompilationMode.NONE
        assert vllm_cfg.cache_config.mamba_cache_mode == "none"
        assert vllm_cfg.cache_config.mamba_block_size == 128

        with (
            set_current_vllm_config(vllm_cfg),
            set_vllm_model_wrapper_context(
                mesh=SimpleNamespace(shape={"model": 1}, axis_names=("model",)),
                vllm_config=vllm_cfg,
            ),
            patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.pallas.jax_op",
                return_value=MagicMock(),
            ),
        ):
            model = Mamba3ForCausalLM(vllm_config=vllm_cfg)
            spec = model.backbone.layers[0].mixer.get_kv_cache_spec(vllm_cfg)

        assert isinstance(spec, MambaSpec)
        assert spec.shapes == Mamba3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_cfg
        )
        assert spec.dtypes == Mamba3ForCausalLM.get_mamba_state_dtype_from_config(
            vllm_cfg
        )

        # Prefix caching enabled must fail fast on startup.
        with pytest.raises(NotImplementedError, match=r"Prefix caching.*not yet"):
            VllmConfig(
                model_config=model_cfg,
                cache_config=CacheConfig(enable_prefix_caching=True),
                device_config=DeviceConfig(device="cpu"),
            )

    def test_state_shape_and_dtype_match_mixer_layer(self):
        for is_mimo in (False, True):
            cfg = Mamba3Config(
                d_model=64,
                n_layer=2,
                vocab_size=256,
                d_intermediate=112 if is_mimo else 128,
                ssm_cfg={
                    "layer": "Mamba3",
                    "d_state": 16,
                    "expand": 2,
                    "headdim": 16,
                    "ngroups": 1,
                    "rope_fraction": 0.5,
                    "is_mimo": is_mimo,
                    "mimo_rank": 4,
                },
            )
            model, vllm_cfg = _build_mamba3_model(cfg)
            mixer = model.backbone.layers[0].mixer
            assert isinstance(mixer, VllmMambaMixer3)
            assert (
                Mamba3ForCausalLM.get_mamba_state_shape_from_config(vllm_cfg)
                == mixer.get_state_shape()
            )
            assert (
                Mamba3ForCausalLM.get_mamba_state_dtype_from_config(vllm_cfg)
                == mixer.get_state_dtype()
            )

        assert SupportsMambaPrefixCaching not in Mamba3ForCausalLM.__mro__
        assert not issubclass(Mamba3ForCausalLM, SupportsMambaPrefixCaching)
        assert not hasattr(Mamba3ForCausalLM, "get_mamba_state_copy_func")


class TestMamba3ArchitectureAndWeights:
    def test_decoder_layer_variants_org_vocab_slicing_and_quant_rejection(self):
        cfg_no_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=100,
            pad_vocab_size_multiple=16,
            d_intermediate=0,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
            },
        )
        model_no_mlp, vllm_cfg = _build_mamba3_model(cfg_no_mlp)
        assert model_no_mlp.backbone.layers[0].mlp is None
        assert model_no_mlp.backbone.layers[0].norm2 is None
        assert not hasattr(model_no_mlp.backbone.layers[0], "residual_in_fp32")
        assert cfg_no_mlp.org_vocab_size == 100
        assert cfg_no_mlp.vocab_size == 112
        assert model_no_mlp.backbone.embedding.org_vocab_size == 112
        logits = model_no_mlp.compute_logits(torch.randn(5, 64))
        assert logits is not None and logits.shape == (5, 100)

        # Quantized weights must raise NotImplementedError cleanly across Mamba-3.
        fake_quant = MagicMock()
        with set_current_vllm_config(vllm_cfg):
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                MambaMixer3(
                    hidden_size=64,
                    ssm_state_size=16,
                    intermediate_size=128,
                    num_heads=8,
                    head_dim=16,
                    quant_config=fake_quant,
                )
            with pytest.raises(NotImplementedError, match="Quantized weights"):
                Mamba3GatedMLP(
                    hidden_size=64,
                    intermediate_size=128,
                    quant_config=fake_quant,
                )

        cfg_mimo_mlp = Mamba3Config(
            d_model=64,
            n_layer=2,
            vocab_size=128,
            d_intermediate=112,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 16,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model_mimo, _ = _build_mamba3_model(cfg_mimo_mlp)
        layer0 = model_mimo.backbone.layers[0]
        assert isinstance(layer0.mlp, Mamba3GatedMLP)
        assert layer0.mlp.orig_intermediate_size == 112
        assert layer0.mlp.intermediate_size == 128
        assert layer0.mixer.mimo_x is layer0.mixer.Psi
        assert layer0.mixer.mimo_o is layer0.mixer.Phi
        assert layer0.mixer.mimo_z is layer0.mixer.Zeta

    def test_load_weights_siso_and_raises_on_unknown_checkpoint_tensor(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=64,
            tie_embeddings=False,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
                "mimo_rank": 4,
                "is_outproj_norm": False,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        total_proj = mixer.total_proj_size

        emb_w = torch.randn(64, 32)
        lm_head_w = torch.randn(64, 32)
        b_bias_3d = torch.randn(4, 1, 8)
        c_bias_3d = torch.randn(4, 1, 8)

        ckpt = [
            ("backbone.embedding.weight", emb_w),
            ("backbone.layers.0.norm.weight", torch.randn(32)),
            ("backbone.layers.0.mixer.in_proj.weight", torch.randn(total_proj, 32)),
            ("backbone.layers.0.mixer.out_proj.weight", torch.randn(32, 64)),
            ("backbone.layers.0.mixer.B_norm.weight", torch.randn(8)),
            ("backbone.layers.0.mixer.C_norm.weight", torch.randn(8)),
            ("backbone.layers.0.mixer.B_bias", b_bias_3d),
            ("backbone.layers.0.mixer.C_bias", c_bias_3d),
            ("backbone.layers.0.mixer.D", torch.randn(4)),
            ("backbone.layers.0.mixer.dt_bias", torch.randn(4)),
            ("backbone.layers.0.norm2.weight", torch.randn(32)),
            ("backbone.layers.0.mlp.fc1.weight", torch.randn(128, 32)),
            ("backbone.layers.0.mlp.fc2.weight", torch.randn(32, 64)),
            ("backbone.norm_f.weight", torch.randn(32)),
            ("lm_head.weight", lm_head_w),
        ]
        loaded = model.load_weights(ckpt)
        assert "backbone.embedding.weight" in loaded
        assert "lm_head.weight" in loaded
        torch.testing.assert_close(mixer.B_bias, b_bias_3d.squeeze(1))
        torch.testing.assert_close(mixer.C_bias, c_bias_3d.squeeze(1))
        torch.testing.assert_close(model.lm_head.weight, lm_head_w)

        with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
            model.load_weights(
                [("backbone.layers.0.mixer.unknown_param", torch.ones(4))]
            )

    def test_pipeline_parallel_missing_layers_and_tied_last_rank_weights(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=2,
            vocab_size=64,
            d_intermediate=0,
            tie_embeddings=True,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
            },
        )
        pp_first = SimpleNamespace(is_first_rank=True, is_last_rank=False)
        pp_last = SimpleNamespace(is_first_rank=False, is_last_rank=True)

        model_first, _ = _build_mamba3_model(
            cfg, pp_group=pp_first, pp_layer_range=(0, 1)
        )
        assert not isinstance(model_first.backbone.embedding, PPMissingLayer)
        assert isinstance(model_first.backbone.layers[1], PPMissingLayer)
        assert isinstance(model_first.backbone.norm_f, PPMissingLayer)
        assert isinstance(model_first.lm_head, PPMissingLayer)
        assert model_first.compute_logits(torch.zeros(2, 32)) is None

        model_last, _ = _build_mamba3_model(
            cfg, pp_group=pp_last, pp_layer_range=(1, 2)
        )
        assert isinstance(model_last.backbone.embedding, PPMissingLayer)
        assert isinstance(model_last.backbone.layers[0], PPMissingLayer)
        assert not isinstance(model_last.backbone.norm_f, PPMissingLayer)
        assert not isinstance(model_last.lm_head, PPMissingLayer)

        emb_w = torch.randn(64, 32)
        d0_w = torch.randn(4)
        d1_w = torch.randn(4)
        norm_f_w = torch.randn(32)
        shared_ckpt = [
            ("backbone.embedding.weight", emb_w),
            ("backbone.layers.0.mixer.D", d0_w),
            ("backbone.layers.1.mixer.D", d1_w),
            ("backbone.norm_f.weight", norm_f_w),
        ]

        loaded_first = model_first.load_weights(shared_ckpt)
        assert "backbone.embedding.weight" in loaded_first
        assert "backbone.layers.0.mixer.D" in loaded_first
        assert "backbone.layers.1.mixer.D" not in loaded_first

        loaded_last = model_last.load_weights(shared_ckpt)
        assert "lm_head.weight" in loaded_last
        assert "backbone.layers.1.mixer.D" in loaded_last
        torch.testing.assert_close(model_last.lm_head.weight, emb_w)

        # Forward handoff from PP rank 0 -> PP rank 1 via IntermediateTensors.
        with patch(
            "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
            return_value=pp_first,
        ):
            inter = model_first(
                torch.tensor([1, 2], dtype=torch.long),
                torch.tensor([0, 1], dtype=torch.long),
            )
        assert isinstance(inter, IntermediateTensors)
        with patch(
            "vllm_torchtpu.models.vllm.mamba3.mamba3.get_pp_group",
            return_value=pp_last,
        ):
            out_last = model_last(
                None,
                torch.tensor([0, 1], dtype=torch.long),
                intermediate_tensors=inter,
            )
        assert isinstance(out_last, torch.Tensor) and out_last.shape == (2, 32)

    def test_load_weights_mimo_with_unpadded_and_prepadded_mlp(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=48,
            tie_embeddings=True,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": True,
                "mimo_rank": 4,
            },
        )
        model, _ = _build_mamba3_model(cfg)
        mixer = model.backbone.layers[0].mixer
        mlp = model.backbone.layers[0].mlp
        assert mlp is not None and mlp.intermediate_size == 64

        psi_w = torch.randn(4, 4, 16)
        phi_w = torch.randn(4, 4, 16)
        zeta_w = torch.randn(4, 4, 16)
        fc1_unpadded = torch.randn(96, 32)
        fc2_unpadded = torch.randn(32, 48)

        model.load_weights(
            [
                ("model.embed_tokens.weight", torch.ones(64, 32)),
                ("model.layers.0.mixer.mimo_x", psi_w),
                ("model.layers.0.mixer.mimo_o", phi_w),
                ("model.layers.0.mixer.mimo_z", zeta_w),
                ("model.layers.0.mlp_norm.weight", torch.ones(32) * 2.0),
                ("model.layers.0.mlp.fc1.weight", fc1_unpadded),
                ("model.layers.0.mlp.fc2.weight", fc2_unpadded),
            ]
        )
        torch.testing.assert_close(mixer.Psi, psi_w)
        torch.testing.assert_close(mixer.Phi, phi_w)
        torch.testing.assert_close(mixer.Zeta, zeta_w)
        torch.testing.assert_close(mlp.fc1.weight[:48], fc1_unpadded[:48])
        assert torch.all(mlp.fc1.weight[48:64] == 0)
        torch.testing.assert_close(mlp.fc1.weight[64:112], fc1_unpadded[48:96])
        assert torch.all(mlp.fc1.weight[112:128] == 0)
        torch.testing.assert_close(mlp.fc2.weight[:, :48], fc2_unpadded)
        assert torch.all(mlp.fc2.weight[:, 48:64] == 0)

    def test_multiprocess_tp_sharding_with_single_group(self):
        cfg = Mamba3Config(
            d_model=32,
            n_layer=1,
            vocab_size=64,
            d_intermediate=48,
            ssm_cfg={
                "layer": "Mamba3",
                "d_state": 8,
                "expand": 2,
                "headdim": 16,
                "ngroups": 1,
                "rope_fraction": 0.5,
                "is_mimo": False,
            },
        )
        model_r1, _ = _build_mamba3_model(cfg, tp_size=2, tp_rank=1)
        mixer_r1 = model_r1.backbone.layers[0].mixer
        mlp_r1 = model_r1.backbone.layers[0].mlp
        assert mlp_r1 is not None
        assert mixer_r1.in_proj.weight.shape == (88, 32)

        in_proj_global = torch.arange(158 * 32, dtype=torch.float32).reshape(158, 32)
        b_bias_3d = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 1, 8)
        fc1_padded = torch.arange(128 * 32, dtype=torch.float32).reshape(128, 32)
        fc2_padded = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)

        model_r1.load_weights(
            [
                ("backbone.layers.0.mixer.in_proj.weight", in_proj_global),
                ("backbone.layers.0.mixer.B_bias", b_bias_3d),
                ("backbone.layers.0.mlp.fc1.weight", fc1_padded),
                ("backbone.layers.0.mlp.fc2.weight", fc2_padded),
            ]
        )

        expected_in_proj_r1 = torch.cat(
            [
                in_proj_global[32:64],
                in_proj_global[96:128],
                in_proj_global[128:136],
                in_proj_global[136:144],
                in_proj_global[146:148],
                in_proj_global[150:152],
                in_proj_global[154:156],
                in_proj_global[156:158],
            ],
            dim=0,
        )
        torch.testing.assert_close(mixer_r1.in_proj.weight, expected_in_proj_r1)
        torch.testing.assert_close(mixer_r1.B_bias, b_bias_3d.squeeze(1)[2:4])
        expected_fc1_r1 = torch.cat([fc1_padded[32:64], fc1_padded[96:128]], dim=0)
        torch.testing.assert_close(mlp_r1.fc1.weight, expected_fc1_r1)
        torch.testing.assert_close(mlp_r1.fc2.weight, fc2_padded[:, 32:64])


class TestMamba3EndToEndForward:
    @pytest.mark.parametrize("is_mimo", [False, True])
    def test_checkpoint_load_and_prefill_decode_match_pytorch_reference(
        self, tmp_path, is_mimo: bool
    ):
        """Loads a synthetic `state-spaces/mamba3-*` safetensors checkpoint and
        verifies prompt logits, greedy tokens, and split prefill+decode continuity
        against the standalone PyTorch Mamba-3 oracle."""
        register_models()
        ckpt_dir = tmp_path / f"mamba3_ckpt_{int(is_mimo)}"
        write_synthetic_mamba3_checkpoint(ckpt_dir, is_mimo=is_mimo, seed=4100)

        prompt_ids = [4, 11, 27, 42, 19]
        gen_steps = 3
        ref_prompt_logits, ref_greedy = eval_torch_mamba3_reference(
            ckpt_dir, prompt_ids, gen_steps=gen_steps
        )

        hf_cfg = get_config(str(ckpt_dir), trust_remote_code=False)
        assert isinstance(hf_cfg, Mamba3Config)
        model, vllm_cfg = _build_mamba3_model(hf_cfg, use_real_jax_op=True)
        sd = load_file(str(ckpt_dir / "model.safetensors"))
        model.load_weights(list(sd.items()))

        aux_shape, ssm_shape = Mamba3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_cfg
        )
        prefixes = [f"backbone.layers.{i}.mixer" for i in range(hf_cfg.n_layer)]
        for layer in model.backbone.layers:
            layer.mixer.kv_cache = (
                torch.zeros((2, *aux_shape), dtype=torch.float32),
                torch.zeros((2, *ssm_shape), dtype=torch.float32),
            )

        # Step 0: Prefill prompt
        meta_pre = make_mamba_attn_metadata(
            mamba_state_indices=torch.tensor([0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, len(prompt_ids)], dtype=torch.int32),
            request_distribution=torch.tensor([0, 1, 1], dtype=torch.int32),
            seq_lens=torch.tensor([len(prompt_ids)], dtype=torch.int32),
        )
        with patch(
            "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
            return_value=make_forward_context(prefixes, meta_pre),
        ):
            hidden_pre = model(
                torch.tensor(prompt_ids, dtype=torch.long),
                torch.arange(len(prompt_ids), dtype=torch.long),
            )
            logits_pre = model.compute_logits(hidden_pre)

        assert logits_pre is not None
        assert logits_pre.shape == (len(prompt_ids), 100)
        torch.testing.assert_close(
            logits_pre[-1], ref_prompt_logits, rtol=1e-4, atol=1e-4
        )

        greedy_ids = [int(torch.argmax(logits_pre[-1]).item())]
        for step in range(1, gen_steps):
            cur_len = len(prompt_ids) + step
            meta_dec = make_mamba_attn_metadata(
                mamba_state_indices=torch.tensor([0], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                request_distribution=torch.tensor([1, 1, 1], dtype=torch.int32),
                seq_lens=torch.tensor([cur_len], dtype=torch.int32),
            )
            with patch(
                "vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils.get_forward_context",
                return_value=make_forward_context(prefixes, meta_dec),
            ):
                h_dec = model(
                    torch.tensor([greedy_ids[-1]], dtype=torch.long),
                    torch.tensor([cur_len - 1], dtype=torch.long),
                )
                logits_dec = model.compute_logits(h_dec)
            assert logits_dec is not None
            greedy_ids.append(int(torch.argmax(logits_dec[-1]).item()))

        assert greedy_ids == ref_greedy


def _serve_checkpoint_in_spawned_worker(
    ckpt_dir_str: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run `vllm.LLM` inside an isolated spawned child so libtpu is released on exit."""
    try:
        import os

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        llm = LLM(
            model=ckpt_dir_str,
            tokenizer=ckpt_dir_str,
            max_model_len=128,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            kv_cache_memory_bytes=256 << 20,
            enable_prefix_caching=False,
            enforce_eager=False,
            async_scheduling=False,
            dtype="bfloat16",
            disable_log_stats=True,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)], params
        )
        out0 = outputs[0].outputs[0]
        first_tok = int(out0.token_ids[0])
        assert out0.logprobs is not None and first_tok in out0.logprobs[0]
        step0_top1_logprob = float(out0.logprobs[0][first_tok].logprob)
        result_queue.put(("ok", (list(out0.token_ids), step0_top1_logprob)))
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        import traceback

        result_queue.put(("err", f"{exc}\n{traceback.format_exc()}"))


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_e2e_vllm_llm_serving_mamba2_and_mamba3_parity_tpu(tmp_path: Path):
    """End-to-end `vllm.LLM` TPU serving parity for Mamba-2 and Mamba-3 checkpoints."""
    ctx = multiprocessing.get_context("spawn")
    prompt_ids = [4, 9, 15, 22, 31, 12]
    max_tokens = 4

    # 1. Mamba-2 E2E vs HuggingFace `transformers.Mamba2ForCausalLM`
    mamba2_dir = tmp_path / "mamba2_e2e"
    write_synthetic_mamba2_checkpoint(mamba2_dir, seed=7001)
    ref_m2_logits, expected_mamba2_tokens = eval_hf_mamba2_reference(
        mamba2_dir, prompt_ids, gen_steps=max_tokens
    )
    q2 = ctx.Queue()
    p2 = ctx.Process(
        target=_serve_checkpoint_in_spawned_worker,
        args=(str(mamba2_dir), prompt_ids, max_tokens, q2),
    )
    p2.start()
    p2.join(timeout=300)
    assert p2.exitcode == 0
    status2, payload2 = q2.get(timeout=10)
    assert status2 == "ok", payload2
    m2_tokens, m2_step0_lp = payload2
    assert m2_tokens == expected_mamba2_tokens
    ref_m2_step0_lp = float(
        torch.log_softmax(ref_m2_logits.float(), dim=-1)[
            expected_mamba2_tokens[0]
        ].item()
    )
    assert abs(m2_step0_lp - ref_m2_step0_lp) < 0.15

    # 2. Mamba-3 SISO & MIMO E2E vs standalone PyTorch Mamba-3 reference
    for is_mimo in (False, True):
        mamba3_dir = tmp_path / f"mamba3_e2e_{int(is_mimo)}"
        write_synthetic_mamba3_checkpoint(mamba3_dir, is_mimo=is_mimo, seed=8001)
        ref_m3_logits, expected_mamba3_tokens = eval_torch_mamba3_reference(
            mamba3_dir, prompt_ids, gen_steps=max_tokens
        )
        q3 = ctx.Queue()
        p3 = ctx.Process(
            target=_serve_checkpoint_in_spawned_worker,
            args=(str(mamba3_dir), prompt_ids, max_tokens, q3),
        )
        p3.start()
        p3.join(timeout=300)
        assert p3.exitcode == 0
        status3, payload3 = q3.get(timeout=10)
        assert status3 == "ok", payload3
        m3_tokens, m3_step0_lp = payload3
        assert m3_tokens == expected_mamba3_tokens
        ref_m3_step0_lp = float(
            torch.log_softmax(ref_m3_logits.float(), dim=-1)[
                expected_mamba3_tokens[0]
            ].item()
        )
        assert abs(m3_step0_lp - ref_m3_step0_lp) < 0.15
