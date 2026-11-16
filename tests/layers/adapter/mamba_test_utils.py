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
"""Shared Mamba-2 and Mamba-3 test fixtures, JAX op bridges, and reference helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax.sharding import Mesh
from safetensors.torch import load_file, save_file


def jax_cpu_mesh() -> Mesh:
    """Return a single-device JAX `Mesh` lazily inside test execution."""
    devices = np.asarray(jax.devices()[:1]).reshape((1,))
    return Mesh(devices, axis_names=("model",))


def passthrough_jax_op(
    op_name: str,
    fn: Any,
    donate_argnums: tuple[int, ...] = (),
) -> Any:
    """CPU test bridge matching `pallas.jax_op` for arbitrary donated state arity."""
    del op_name, donate_argnums

    def _op(*torch_args: Any) -> tuple[torch.Tensor, ...]:
        jax_args = [
            jnp.asarray(t.detach().cpu().to(torch.float32).numpy()).astype(
                jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
            )
            if isinstance(t, torch.Tensor) and t.is_floating_point()
            else (
                jnp.asarray(t.detach().cpu().numpy())
                if isinstance(t, torch.Tensor)
                else None
            )
            for t in torch_args
        ]
        results = fn(*jax_args)
        out_tensors: list[torch.Tensor] = []
        for item in results:
            if item.dtype == jnp.bfloat16:
                arr_f32 = np.array(item.astype(jnp.float32), copy=True)
                out_tensors.append(torch.from_numpy(arr_f32).to(torch.bfloat16))
            else:
                out_tensors.append(torch.from_numpy(np.array(item, copy=True)))
        return tuple(out_tensors)

    _op.register_fake = lambda _: None
    return _op


def make_mamba_attn_metadata(
    *,
    mamba_state_indices: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor | None = None,
    mamba_request_distribution: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Build mock `MambaAttentionMetadata` with all declared protocol attributes."""
    return SimpleNamespace(
        mamba_state_indices=mamba_state_indices,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
        mamba_request_distribution=mamba_request_distribution,
        seq_lens=seq_lens,
        block_tables=block_tables,
    )


def make_forward_context(
    prefixes: Sequence[str],
    metadata: SimpleNamespace,
) -> SimpleNamespace:
    """Build a mock `ForwardContext` mapping each layer prefix to `metadata`."""
    return SimpleNamespace(attn_metadata={p: metadata for p in prefixes})


def write_minimal_fast_tokenizer(ckpt_dir: Path, vocab_size: int) -> None:
    """Write a deterministic offline `PreTrainedTokenizerFast` into `ckpt_dir`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for idx in range(4, vocab_size):
        vocab[f"tok{idx}"] = idx

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tok.save_pretrained(str(ckpt_dir))


def write_synthetic_mamba2_checkpoint(
    ckpt_dir: Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create a deterministic HF `Mamba2ForCausalLM` safetensors checkpoint."""
    from transformers import Mamba2Config, Mamba2ForCausalLM

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = Mamba2Config(
        vocab_size=128,
        hidden_size=64,
        state_size=16,
        num_hidden_layers=2,
        head_dim=16,
        num_heads=8,
        n_groups=2,
        expand=2,
        conv_kernel=4,
        chunk_size=8,
        tie_word_embeddings=False,
        use_conv_bias=True,
        use_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    hf_model = Mamba2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in hf_model.named_parameters():
            if "A_log" in name:
                p.copy_(torch.linspace(-0.5, 0.2, p.numel()).reshape_as(p))
            elif "norm" in name or name.endswith(".D"):
                p.copy_(torch.linspace(0.8, 1.2, p.numel()).reshape_as(p))
            elif "dt_bias" in name:
                p.zero_()
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.06)

    hf_model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    write_minimal_fast_tokenizer(ckpt_dir, cfg.vocab_size)
    return {"vocab_size": cfg.vocab_size}


@torch.no_grad()
def eval_hf_mamba2_reference(
    ckpt_dir: Path,
    prompt_token_ids: list[int],
    gen_steps: int,
) -> tuple[torch.Tensor, list[int]]:
    """Run HF `Mamba2ForCausalLM` CPU reference (`(last_prompt_logits, greedy_ids)`)."""
    from transformers import Mamba2ForCausalLM

    hf_model = Mamba2ForCausalLM.from_pretrained(str(ckpt_dir)).to(torch.float32).eval()
    seq = list(prompt_token_ids)
    prompt_logits: torch.Tensor | None = None
    generated: list[int] = []

    for step in range(gen_steps):
        inp = torch.tensor([seq], dtype=torch.long)
        logits = hf_model(input_ids=inp).logits[0, -1].detach().cpu()
        if step == 0:
            prompt_logits = logits.clone()
        next_tok = int(torch.argmax(logits).item())
        generated.append(next_tok)
        seq.append(next_tok)

    assert prompt_logits is not None
    return prompt_logits, generated


def write_synthetic_mamba3_checkpoint(
    ckpt_dir: Path,
    *,
    is_mimo: bool = False,
    seed: int = 3030,
) -> dict[str, Any]:
    """Write a deterministic raw `state-spaces/mamba3-*` safetensors checkpoint.

    Intentionally omits `model_type` and `architectures` in `config.json` (using
    `ssm_cfg.layer = "Mamba3"`) and uses unaligned `vocab_size = 100` with
    `pad_vocab_size_multiple = 16` (`112` stored rows) to exercise
    `_Mamba3HFConfigParser` and `LogitsProcessor(..., org_vocab_size=100)`.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + int(is_mimo))

    d_model = 64
    n_layer = 2
    org_vocab_size = 100
    pad_mult = 16
    padded_vocab_size = 112
    d_intermediate = 64
    d_state = 16
    expand = 2
    d_inner = d_model * expand
    head_dim = 16
    num_heads = d_inner // head_dim
    n_groups = 1
    rope_fraction = 0.5
    num_rope_angles = int((d_state // 2) * rope_fraction)
    mimo_rank = 4 if is_mimo else 1
    chunk_size = 8
    use_outproj_norm = not is_mimo

    config_json = {
        "d_model": d_model,
        "d_intermediate": d_intermediate,
        "n_layer": n_layer,
        "vocab_size": org_vocab_size,
        "ssm_cfg": {
            "layer": "Mamba3",
            "d_state": d_state,
            "expand": expand,
            "headdim": head_dim,
            "ngroups": n_groups,
            "rope_fraction": rope_fraction,
            "A_floor": 1e-4,
            "chunk_size": chunk_size,
            "is_mimo": is_mimo,
            "mimo_rank": 4,
            "is_outproj_norm": use_outproj_norm,
        },
        "rms_norm": True,
        "residual_in_fp32": True,
        "tie_embeddings": True,
        "pad_vocab_size_multiple": pad_mult,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    total_proj = (
        2 * d_inner
        + 2 * mimo_rank * n_groups * d_state
        + 3 * num_heads
        + num_rope_angles
    )
    tensors: dict[str, torch.Tensor] = {
        "backbone.embedding.weight": torch.randn(padded_vocab_size, d_model) * 0.08,
        "backbone.norm_f.weight": torch.linspace(0.85, 1.15, d_model),
    }
    for layer_idx in range(n_layer):
        p = f"backbone.layers.{layer_idx}"
        tensors[f"{p}.norm.weight"] = torch.linspace(0.9, 1.1, d_model)
        tensors[f"{p}.mixer.in_proj.weight"] = torch.randn(total_proj, d_model) * 0.06
        tensors[f"{p}.mixer.out_proj.weight"] = torch.randn(d_model, d_inner) * 0.06
        tensors[f"{p}.mixer.B_norm.weight"] = torch.linspace(0.8, 1.2, d_state)
        tensors[f"{p}.mixer.C_norm.weight"] = torch.linspace(0.8, 1.2, d_state)
        tensors[f"{p}.mixer.B_bias"] = torch.randn(num_heads, mimo_rank, d_state) * 0.05
        tensors[f"{p}.mixer.C_bias"] = torch.randn(num_heads, mimo_rank, d_state) * 0.05
        tensors[f"{p}.mixer.D"] = torch.linspace(0.5, 1.0, num_heads)
        tensors[f"{p}.mixer.b_dt"] = torch.zeros(num_heads)
        if use_outproj_norm:
            tensors[f"{p}.mixer.norm.weight"] = torch.linspace(0.9, 1.1, d_inner)
        if is_mimo:
            tensors[f"{p}.mixer.mimo_x"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.2
            )
            tensors[f"{p}.mixer.mimo_o"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.2
            )
            tensors[f"{p}.mixer.mimo_z"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.3
            )
        tensors[f"{p}.mlp_norm.weight"] = torch.linspace(0.9, 1.1, d_model)
        tensors[f"{p}.mlp.fc1.weight"] = torch.randn(2 * d_intermediate, d_model) * 0.06
        tensors[f"{p}.mlp.fc2.weight"] = torch.randn(d_model, d_intermediate) * 0.06

    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    write_minimal_fast_tokenizer(ckpt_dir, org_vocab_size)
    return {
        "org_vocab_size": org_vocab_size,
        "padded_vocab_size": padded_vocab_size,
    }


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


@torch.no_grad()
def eval_torch_mamba3_reference(
    ckpt_dir: Path,
    prompt_token_ids: list[int],
    gen_steps: int,
) -> tuple[torch.Tensor, list[int]]:
    """Pure-PyTorch step-by-step Mamba-3 oracle (CUDA-free `mamba_ssm` equivalent)."""
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    sd = {
        k: v.float() for k, v in load_file(str(ckpt_dir / "model.safetensors")).items()
    }

    d_model = int(cfg["d_model"])
    n_layer = int(cfg["n_layer"])
    org_vocab_size = int(cfg["vocab_size"])
    d_inter = int(cfg["d_intermediate"])
    ssm = cfg["ssm_cfg"]
    d_state = int(ssm["d_state"])
    expand = int(ssm["expand"])
    d_inner = d_model * expand
    head_dim = int(ssm["headdim"])
    num_heads = d_inner // head_dim
    n_groups = int(ssm["ngroups"])
    group_rep = num_heads // n_groups
    is_mimo = bool(ssm["is_mimo"])
    r = int(ssm["mimo_rank"]) if is_mimo else 1
    num_rope_angles = int((d_state // 2) * float(ssm["rope_fraction"]))
    a_floor = float(ssm.get("A_floor", 1e-4))
    use_outproj_norm = bool(ssm.get("is_outproj_norm", False))
    eps = 1e-5

    def _forward_full(tokens: list[int]) -> torch.Tensor:
        inp = torch.tensor(tokens, dtype=torch.long)
        hidden = sd["backbone.embedding.weight"][inp]
        residual: torch.Tensor | None = None

        s1 = d_inner
        s2 = s1 + d_inner
        s3 = s2 + r * n_groups * d_state
        s4 = s3 + r * n_groups * d_state
        s5 = s4 + num_heads
        s6 = s5 + num_heads
        s7 = s6 + num_heads

        for layer_idx in range(n_layer):
            p = f"backbone.layers.{layer_idx}"
            if residual is None:
                residual = hidden
                hidden = _rms_norm_torch(hidden, sd[f"{p}.norm.weight"], eps)
            else:
                residual = hidden + residual
                hidden = _rms_norm_torch(residual, sd[f"{p}.norm.weight"], eps)

            proj = hidden @ sd[f"{p}.mixer.in_proj.weight"].T
            b_norm_w = sd[f"{p}.mixer.B_norm.weight"]
            c_norm_w = sd[f"{p}.mixer.C_norm.weight"]
            b_bias = sd[f"{p}.mixer.B_bias"]
            c_bias = sd[f"{p}.mixer.C_bias"]
            d_skip = sd[f"{p}.mixer.D"]
            dt_bias = sd[f"{p}.mixer.b_dt"]
            psi = sd.get(f"{p}.mixer.mimo_x")
            phi = sd.get(f"{p}.mixer.mimo_o")
            zeta = sd.get(f"{p}.mixer.mimo_z")
            norm_w = sd.get(f"{p}.mixer.norm.weight") if use_outproj_norm else None

            h = torch.zeros(num_heads, head_dim, d_state)
            prev_x = torch.zeros(num_heads, r, head_dim)
            prev_k = torch.zeros(num_heads, r, d_state)
            theta = torch.zeros(num_heads, num_rope_angles)
            y_rows: list[torch.Tensor] = []

            for t in range(proj.shape[0]):
                row = proj[t]
                z_raw = row[:s1]
                v_t = row[s1:s2].reshape(num_heads, head_dim)
                b_raw = row[s2:s3].reshape(n_groups, r, d_state)
                c_raw = row[s3:s4].reshape(n_groups, r, d_state)
                dd_dt = row[s4:s5]
                dd_a = row[s5:s6]
                trap_raw = row[s6:s7]
                angle_raw = row[s7:]

                b_exp = _rms_norm_torch(b_raw, b_norm_w, eps).repeat_interleave(
                    group_rep, dim=0
                )
                c_exp = _rms_norm_torch(c_raw, c_norm_w, eps).repeat_interleave(
                    group_rep, dim=0
                )
                k_pre = b_exp + b_bias
                q_pre = c_exp + c_bias

                a_t = torch.clamp(-F.softplus(dd_a), max=-a_floor)
                dt_t = F.softplus(dd_dt + dt_bias)
                trap_t = torch.sigmoid(trap_raw)
                gamma_t = dt_t * trap_t
                beta_t = dt_t * (1.0 - trap_t)

                theta = torch.remainder(
                    theta
                    + torch.tanh(angle_raw).unsqueeze(0) * math.pi * dt_t.unsqueeze(1),
                    2.0 * math.pi,
                )
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)

                k_rot = k_pre.clone()
                q_rot = q_pre.clone()
                if is_mimo:
                    assert psi is not None and phi is not None and zeta is not None
                    half = d_state // 2
                    for arr in (k_rot, q_rot):
                        a0 = arr[:, :, :num_rope_angles].clone()
                        a1 = arr[:, :, half : half + num_rope_angles].clone()
                        arr[:, :, :num_rope_angles] = a0 * cos_t.unsqueeze(
                            1
                        ) - a1 * sin_t.unsqueeze(1)
                        arr[:, :, half : half + num_rope_angles] = a0 * sin_t.unsqueeze(
                            1
                        ) + a1 * cos_t.unsqueeze(1)
                    x_t = v_t.unsqueeze(1) * psi
                else:
                    rot_dim = 2 * num_rope_angles
                    for arr in (k_rot, q_rot):
                        pairs = arr[:, 0, :rot_dim].reshape(
                            num_heads, num_rope_angles, 2
                        )
                        p0 = pairs[:, :, 0].clone()
                        p1 = pairs[:, :, 1].clone()
                        arr[:, 0, :rot_dim] = torch.stack(
                            [p0 * cos_t - p1 * sin_t, p0 * sin_t + p1 * cos_t], dim=-1
                        ).reshape(num_heads, rot_dim)
                    x_t = v_t.unsqueeze(1)

                prev_outer = (
                    torch.einsum("hrp,hrn->hpn", prev_x, prev_k)
                    if t > 0
                    else torch.zeros_like(h)
                )
                cur_outer = torch.einsum("hrp,hrn->hpn", x_t, k_rot)
                h = (
                    torch.exp(a_t * dt_t)[:, None, None]
                    * (h + beta_t[:, None, None] * prev_outer)
                    + gamma_t[:, None, None] * cur_outer
                )
                y_ssm = (
                    torch.einsum("hrn,hpn->hrp", q_rot, h) + d_skip[:, None, None] * x_t
                )
                if is_mimo:
                    assert phi is not None and zeta is not None
                    z_r = z_raw.reshape(num_heads, 1, head_dim)
                    y_tok = (
                        (y_ssm * F.silu(z_r * zeta) * phi).sum(dim=1).reshape(d_inner)
                    )
                    val = y_tok
                else:
                    val = y_ssm[:, 0, :].reshape(d_inner) * F.silu(z_raw)

                if norm_w is not None:
                    grouped = val.reshape(n_groups, d_inner // n_groups)
                    var = grouped.pow(2).mean(dim=-1, keepdim=True)
                    val = (grouped * torch.rsqrt(var + eps)).reshape(d_inner) * norm_w
                y_rows.append(val)
                prev_x = x_t.clone()
                prev_k = k_rot.clone()

            hidden = torch.stack(y_rows, dim=0) @ sd[f"{p}.mixer.out_proj.weight"].T

            if d_inter > 0:
                residual = hidden + residual
                hidden = _rms_norm_torch(residual, sd[f"{p}.mlp_norm.weight"], eps)
                gate_up = hidden @ sd[f"{p}.mlp.fc1.weight"].T
                gate, up = gate_up.split(d_inter, dim=-1)
                hidden = (F.silu(gate) * up) @ sd[f"{p}.mlp.fc2.weight"].T

        assert residual is not None
        final_hidden = _rms_norm_torch(
            hidden + residual, sd["backbone.norm_f.weight"], eps
        )
        logits = final_hidden @ sd["backbone.embedding.weight"][:org_vocab_size].T
        return logits

    seq = list(prompt_token_ids)
    prompt_logits: torch.Tensor | None = None
    generated: list[int] = []
    for step in range(gen_steps):
        logits_all = _forward_full(seq)
        last_logits = logits_all[-1].detach().cpu()
        if step == 0:
            prompt_logits = last_logits.clone()
        next_tok = int(torch.argmax(last_logits).item())
        generated.append(next_tok)
        seq.append(next_tok)

    assert prompt_logits is not None
    return prompt_logits, generated
