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
"""Lightweight HF/vLLM configuration and parser registration for Mamba-3."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import AutoConfig, PretrainedConfig
from vllm.model_executor.models.config import MambaModelConfig
from vllm.transformers_utils import config as vllm_tf_config
from vllm.transformers_utils.config import (
    HFConfigParser,
    _maybe_remap_hf_config_attrs,
    register_config_parser,
)
from vllm.transformers_utils.utils import without_trust_remote_code

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_MAMBA3_PARSER_REGISTERED = False


class Mamba3Config(PretrainedConfig):
    """Normalized configuration class for Mamba-3 models (`state-spaces/mamba3-*`)."""

    model_type = "mamba3"

    def __init__(
        self,
        d_model: int | None = None,
        hidden_size: int = 2048,
        d_intermediate: int = 0,
        n_layer: int | None = None,
        num_hidden_layers: int = 24,
        vocab_size: int = 128256,
        ssm_cfg: dict[str, Any] | None = None,
        rms_norm: bool = True,
        norm_epsilon: float | None = None,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        tie_embeddings: bool | None = None,
        tie_word_embeddings: bool = True,
        pad_vocab_size_multiple: int = 16,
        **kwargs: Any,
    ) -> None:
        ssm_cfg_dict = dict(ssm_cfg) if isinstance(ssm_cfg, dict) else {}

        resolved_hidden = int(d_model if d_model is not None else hidden_size)
        resolved_layers = int(n_layer if n_layer is not None else num_hidden_layers)

        raw_vocab_size = int(vocab_size)
        org_vocab_arg = kwargs.pop("org_vocab_size", None)
        self.org_vocab_size = (
            int(org_vocab_arg) if org_vocab_arg is not None else raw_vocab_size
        )
        pad_mult = int(pad_vocab_size_multiple)
        padded_vocab_size = raw_vocab_size
        if pad_mult > 1 and padded_vocab_size % pad_mult != 0:
            padded_vocab_size += pad_mult - (padded_vocab_size % pad_mult)

        self.d_model = resolved_hidden
        self.hidden_size = resolved_hidden
        self.d_intermediate = int(d_intermediate)
        self.n_layer = resolved_layers
        self.num_hidden_layers = resolved_layers
        self.vocab_size = padded_vocab_size
        self.pad_vocab_size_multiple = pad_mult
        self.ssm_cfg = ssm_cfg_dict

        self.state_size = int(
            ssm_cfg_dict.get(
                "d_state",
                kwargs.pop("state_size", kwargs.pop("d_state", 128)),
            )
        )
        self.expand = int(ssm_cfg_dict.get("expand", kwargs.pop("expand", 2)))
        self.head_dim = int(
            ssm_cfg_dict.get(
                "headdim",
                ssm_cfg_dict.get("head_dim", kwargs.pop("head_dim", 64)),
            )
        )
        inter_arg = kwargs.pop("intermediate_size", None)
        self.intermediate_size = (
            int(inter_arg) if inter_arg else (self.expand * self.hidden_size)
        )
        heads_arg = kwargs.pop("num_heads", None)
        self.num_heads = (
            int(heads_arg) if heads_arg else (self.intermediate_size // self.head_dim)
        )
        self.n_groups = int(
            ssm_cfg_dict.get(
                "ngroups",
                ssm_cfg_dict.get("n_groups", kwargs.pop("n_groups", 1)),
            )
        )
        self.is_mimo = bool(ssm_cfg_dict.get("is_mimo", kwargs.pop("is_mimo", False)))
        raw_rank = int(
            ssm_cfg_dict.get(
                "mimo_rank",
                kwargs.pop("mimo_rank", 4 if self.is_mimo else 1),
            )
        )
        self.mimo_rank = raw_rank if self.is_mimo else 1
        self.rope_fraction = float(
            ssm_cfg_dict.get("rope_fraction", kwargs.pop("rope_fraction", 0.5))
        )
        default_chunk = 16 if self.is_mimo else 64
        self.chunk_size = int(
            ssm_cfg_dict.get("chunk_size", kwargs.pop("chunk_size", default_chunk))
        )
        self.A_floor = float(ssm_cfg_dict.get("A_floor", kwargs.pop("A_floor", 1e-4)))
        self.use_rms_norm = bool(
            ssm_cfg_dict.get("is_outproj_norm", kwargs.pop("use_rms_norm", False))
        )
        self.use_bias = bool(ssm_cfg_dict.get("bias", kwargs.pop("use_bias", False)))

        resolved_eps = float(
            kwargs.pop(
                "rms_norm_eps",
                norm_epsilon if norm_epsilon is not None else layer_norm_epsilon,
            )
        )
        self.rms_norm = bool(rms_norm)
        self.norm_epsilon = resolved_eps
        self.layer_norm_epsilon = resolved_eps
        self.rms_norm_eps = resolved_eps
        self.residual_in_fp32 = bool(residual_in_fp32)

        resolved_tie = bool(
            tie_embeddings if tie_embeddings is not None else tie_word_embeddings
        )
        self.tie_embeddings = resolved_tie

        if not kwargs.get("architectures"):
            kwargs["architectures"] = ["Mamba3ForCausalLM"]
        kwargs["tie_word_embeddings"] = resolved_tie
        super().__init__(**kwargs)


def ensure_mamba3_config(config: PretrainedConfig) -> Mamba3Config:
    """Return ``config`` as a normalized ``Mamba3Config``."""
    if isinstance(config, Mamba3Config):
        return config
    return Mamba3Config.from_dict(config.to_dict())


class TPUMamba3Config(MambaModelConfig):
    """Mamba cache initialization and TPU support checks for Mamba3ForCausalLM."""

    @classmethod
    def verify_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().verify_and_update_config(vllm_config)
        cache_config = vllm_config.cache_config
        if cache_config.enable_prefix_caching or cache_config.mamba_cache_mode in (
            "all",
            "align",
        ):
            raise NotImplementedError(
                "Prefix caching is not yet supported for Mamba3ForCausalLM on TPU. "
                "Pass --no-enable-prefix-caching."
            )
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "Quantized weights are not yet supported for Mamba3ForCausalLM on TPU."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Speculative decoding is not yet supported for "
                "Mamba3ForCausalLM on TPU."
            )
        if (
            vllm_config.parallel_config.prefill_context_parallel_size > 1
            or vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Context parallelism is not yet supported for Mamba3ForCausalLM on TPU."
            )


class _Mamba3HFConfigParser(HFConfigParser):
    """Scoped vLLM HF config parser that detects raw `ssm_cfg.layer == "Mamba3"`."""

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], PretrainedConfig]:
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = (
            vllm_tf_config.huggingface_hub.constants.HF_HUB_OFFLINE
        )
        local_kwargs = without_trust_remote_code(local_kwargs)
        hf_overrides = local_kwargs.pop("hf_overrides", None)
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            **local_kwargs,
        )
        ssm_cfg = config_dict.get("ssm_cfg") if isinstance(config_dict, dict) else None
        overridden_type = (
            hf_overrides.get("model_type") if isinstance(hf_overrides, dict) else None
        )
        is_mamba3 = (
            overridden_type == "mamba3"
            or (
                isinstance(config_dict, dict)
                and config_dict.get("model_type") == "mamba3"
            )
            or (
                isinstance(ssm_cfg, dict)
                and ssm_cfg.get("layer") == "Mamba3"
                and overridden_type is None
            )
        )
        if not is_mamba3:
            return super().parse(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs,
            )

        merged = dict(config_dict)
        if isinstance(hf_overrides, dict):
            merged.update(hf_overrides)
        merged.setdefault("model_type", "mamba3")
        merged.setdefault("architectures", ["Mamba3ForCausalLM"])
        local_kwargs.setdefault("name_or_path", str(model))
        config = Mamba3Config.from_dict(merged, **local_kwargs)
        if callable(hf_overrides):
            config = hf_overrides(config)
        config = _maybe_remap_hf_config_attrs(config)
        return merged, config


def register_mamba3_config() -> None:
    """Register `Mamba3Config` with `AutoConfig` and vLLM's `"hf"` config parser."""
    global _MAMBA3_PARSER_REGISTERED
    AutoConfig.register("mamba3", Mamba3Config, exist_ok=True)
    if _MAMBA3_PARSER_REGISTERED:
        return
    register_config_parser("hf")(_Mamba3HFConfigParser)
    _MAMBA3_PARSER_REGISTERED = True
