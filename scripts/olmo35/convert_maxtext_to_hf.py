#!/usr/bin/env python3
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
"""Convert the OLMo 3.5 Small MaxText (Orbax / zarr, fp32) checkpoint to the
HF-style bf16 safetensors that vllm-torchtpu's Olmo35ForCausalLM loads.

Reads each parameter directly with TensorStore (no MaxText import), one shard
per decoder layer plus one for embedding / final norm / LM head. Resumable:
a shard is skipped when it already exists.

Run on a CDK pod with /workspace/venv (tensorstore, torch, safetensors):
  python3 scripts/olmo35/convert_maxtext_to_hf.py \
      --src gs://sierraqian/olmo35/checkpoints/162974/items \
      --dst /workspace/models/olmo35-small
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import time

import numpy as np
import tensorstore as ts
import torch
from safetensors.torch import save_file

NUM_LAYERS = 40
FULL_ATTN_INTERVAL = 8
FIRST_DENSE = 1
NUM_EXPERTS = 256
HEAD_DIM = 256
# Kept in fp32 (tiny, and the model holds them in fp32).
FP32_SUFFIXES = (".A_log", ".dt_bias", ".ssmax_scale")

_CTX = ts.Context(
    {
        "gcs_request_concurrency": {"limit": 256},
        "file_io_concurrency": {"limit": 256},
        "data_copy_concurrency": {"limit": 64},
    }
)


def _spec(src: str, name: str) -> dict:
    path = f"{src.rstrip('/')}/params.params.{name}/"
    if path.startswith("gs://"):
        bucket, _, key = path[len("gs://") :].partition("/")
        kvstore = {"driver": "gcs", "bucket": bucket, "path": key}
    else:
        kvstore = {"driver": "file", "path": path}
    # One chunk of the source checkpoint was repaired by hand; tolerate any
    # other missing chunk as zeros only after logging (see _read).
    return {"driver": "zarr", "kvstore": kvstore}


def _read(src: str, name: str) -> np.ndarray:
    t = ts.open(_spec(src, name), context=_CTX, open=True).result()
    return t.read().result()


def _bf16(x: np.ndarray, name: str) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x))
    if name.endswith(FP32_SUFFIXES):
        return t.float().contiguous()
    return t.to(torch.bfloat16).contiguous()


def _linear(kernel: np.ndarray) -> np.ndarray:
    """MaxText [in, out] kernel -> torch [out, in] weight."""
    return np.ascontiguousarray(kernel.T)


def convert_layer(src: str, i: int) -> dict[str, torch.Tensor]:
    L = f"decoder.layers_{i}"
    P = f"model.layers.{i}"
    out: dict[str, np.ndarray] = {}
    for norm in ("attn_in_norm", "attn_out_norm", "ffn_in_norm", "ffn_out_norm"):
        out[f"{P}.{norm}.weight"] = _read(src, f"{L}.{norm}.scale")

    if (i + 1) % FULL_ATTN_INTERVAL == 0:
        A = f"{L}.mixer.attention"
        q = _read(src, f"{A}.query.kernel")  # [D, H, 2 * hd]: per head [q | gate]
        d, h, _ = q.shape
        out[f"{P}.mixer.q_proj.weight"] = _linear(q[:, :, :HEAD_DIM].reshape(d, -1))
        out[f"{P}.mixer.g_proj.weight"] = _linear(q[:, :, HEAD_DIM:].reshape(d, -1))
        del q
        for proj in ("key", "value"):
            k = _read(src, f"{A}.{proj}.kernel")  # [D, KV, hd]
            out[f"{P}.mixer.{proj[0]}_proj.weight"] = _linear(k.reshape(d, -1))
        o = _read(src, f"{A}.out.kernel")  # [H * hd, D]
        out[f"{P}.mixer.o_proj.weight"] = _linear(o.reshape(-1, d))
        out[f"{P}.mixer.q_norm"] = _read(src, f"{A}.query_norm.scale")
        out[f"{P}.mixer.k_norm"] = _read(src, f"{A}.key_norm.scale")
        out[f"{P}.mixer.ssmax_scale"] = _read(src, f"{A}.ssmax_scale")
    else:
        M = f"{L}.mixer"
        for mt, hf in (
            ("w_q", "q_proj"),
            ("w_k", "k_proj"),
            ("w_v", "v_proj"),
            ("w_b", "b_proj"),
            ("f_proj_1", "f_a_proj"),
            ("f_proj_2", "f_b_proj"),
            ("g_proj_1", "g_a_proj"),
            ("g_proj_2", "g_b_proj"),
            ("w_out", "o_proj"),
        ):
            out[f"{P}.mixer.{hf}.weight"] = _linear(_read(src, f"{M}.{mt}.kernel"))
        out[f"{P}.mixer.g_b_proj.bias"] = _read(src, f"{M}.g_proj_2.bias")
        for c in ("q", "k", "v"):
            w = _read(src, f"{M}.{c}_conv")  # [channels, kernel]
            out[f"{P}.mixer.{c}_conv1d.weight"] = w[:, None, :]
        out[f"{P}.mixer.A_log"] = _read(src, f"{M}.A_log")
        out[f"{P}.mixer.dt_bias"] = _read(src, f"{M}.dt_bias")
        out[f"{P}.mixer.o_norm.weight"] = _read(src, f"{M}.o_norm.scale")

    ffn = f"{P}.mlp" if i < FIRST_DENSE else f"{P}.moe.shared_experts"
    for mt, hf in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
        out[f"{ffn}.{hf}.weight"] = _linear(_read(src, f"{L}.shared_ffn.{mt}.kernel"))

    tensors = {k: _bf16(v, k) for k, v in out.items()}
    del out
    if i >= FIRST_DENSE:
        tensors[f"{P}.moe.gate.weight"] = _bf16(
            _linear(_read(src, f"{L}.moe_block.gate.kernel")), "gate"
        )
        tensors[f"{P}.moe.latent_down.weight"] = _bf16(
            _linear(_read(src, f"{L}.latent_down.kernel")), "ld"
        )
        tensors[f"{P}.moe.latent_up.weight"] = _bf16(
            _linear(_read(src, f"{L}.latent_up.kernel")), "lu"
        )
        for mt, hf in (("wi_0", "gate_proj"), ("wi_1", "up_proj"), ("wo", "down_proj")):
            w = torch.from_numpy(_read(src, f"{L}.moe_block.{mt}")).to(torch.bfloat16)
            assert w.shape[0] == NUM_EXPERTS, (mt, w.shape)
            # wi: [E, latent, F] -> [F, latent]; wo: [E, F, latent] -> [latent, F]
            for e in range(NUM_EXPERTS):
                tensors[f"{P}.moe.experts.{e}.{hf}.weight"] = w[e].t().contiguous()
            del w
    return tensors


def convert_globals(src: str) -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": _bf16(_read(src, "token_embedder.embedding"), "e"),
        "model.embedding_norm.weight": _bf16(
            _read(src, "decoder.embedding_norm.scale"), "n"
        ),
        "model.norm.weight": _bf16(_read(src, "decoder.decoder_norm.scale"), "n"),
        "lm_head.weight": _bf16(_linear(_read(src, "decoder.logits_dense.kernel")), "h"),
    }


_GCS = None


def _gcs_blob(uri: str):
    global _GCS
    from google.cloud import storage

    if _GCS is None:
        _GCS = storage.Client()
    bucket, _, key = uri[len("gs://") :].partition("/")
    return _GCS.bucket(bucket).blob(key)


def _exists(path: str) -> bool:
    return _gcs_blob(path).exists() if path.startswith("gs://") else os.path.exists(path)


def _publish(local_tmp: str, path: str) -> None:
    """Move a finished shard to its destination (local rename or GCS upload)."""
    if path.startswith("gs://"):
        blob = _gcs_blob(path)
        blob.chunk_size = 256 * 1024 * 1024
        blob.upload_from_filename(local_tmp, timeout=3600)
        os.remove(local_tmp)
    else:
        os.replace(local_tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gs://sierraqian/olmo35/checkpoints/162974/items")
    ap.add_argument("--dst", default="/workspace/models/olmo35-small")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--layers", default="", help="e.g. 0,1,7 for a partial run")
    args = ap.parse_args()
    remote = args.dst.startswith("gs://")
    if not remote:
        os.makedirs(args.dst, exist_ok=True)
    tmp_dir = "/tmp/olmo35_shards" if remote else args.dst
    os.makedirs(tmp_dir, exist_ok=True)
    total = NUM_LAYERS + 1
    shard = lambda j: f"model-{j:05d}-of-{total:05d}.safetensors"  # noqa: E731

    jobs = [("globals", 0)] + [("layer", i) for i in range(NUM_LAYERS)]
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        jobs = [j for j in jobs if j[0] == "layer" and j[1] in keep]

    def run(job):
        kind, i = job
        j = 0 if kind == "globals" else i + 1
        path = f"{args.dst.rstrip('/')}/{shard(j)}"
        if _exists(path):
            print(f"[skip] {shard(j)} exists", flush=True)
            return
        t0 = time.time()
        tensors = convert_globals(args.src) if kind == "globals" else convert_layer(args.src, i)
        tmp = os.path.join(tmp_dir, shard(j) + ".tmp")
        save_file(tensors, tmp, metadata={"format": "pt"})
        gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
        del tensors
        _publish(tmp, path)
        print(f"[done] {kind} {i}: {gb:.2f} GB, {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for f in cf.as_completed([pool.submit(run, j) for j in jobs]):
            f.result()
    print(f"converted in {time.time() - t0:.0f}s", flush=True)

    if args.layers:
        return
    # Index over every shard present.
    import struct

    weight_map, size = {}, 0
    for j in range(total):
        path = f"{args.dst.rstrip('/')}/{shard(j)}"
        if remote:
            blob = _gcs_blob(path)
            n = struct.unpack("<Q", blob.download_as_bytes(start=0, end=7))[0]
            header = json.loads(blob.download_as_bytes(start=8, end=8 + n - 1))
        else:
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            weight_map[k] = shard(j)
            size += meta["data_offsets"][1] - meta["data_offsets"][0]
    index = json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map}, indent=1)
    if remote:
        _gcs_blob(f"{args.dst.rstrip('/')}/model.safetensors.index.json").upload_from_string(index)
    else:
        with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
            f.write(index)
    n_expert = sum(1 for k in weight_map if re.search(r"\.experts\.\d+\.", k))
    print(f"index: {len(weight_map)} tensors ({n_expert} expert), {size / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
