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
"""Raiden (tpu_sync) weight synchronization for RL rollouts on TorchTPU."""

from __future__ import annotations

import math
import re
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.dp_vocab_parallel import DP_VOCAB_ATTR
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ws_lib: Any = None
_RAIDEN_IMPORT_ERROR: Exception | None = None

_SKIP_PREFIXES = ("visual", "mtp")
_WRAPPER_ROOTS = ("language_model", "base", "model")
_NAME_REWRITES = ((".experts.routed_experts.", ".experts."),)
_FOLDED_INDEX_RE = re.compile(r"^(.+?)_(\d+)$")
_HEAD_SHARDED_ATTRS = frozenset({"A_log", "dt_bias"})
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "127.0.0.1", "::", "[::]", "[::1]"})


def _get_ws_lib() -> Any:
    """Import tpu_sync lazily on TPU workers so CPU imports skip C++ PJRT init."""
    global _ws_lib, _RAIDEN_IMPORT_ERROR
    if _ws_lib is not None:
        return _ws_lib
    try:
        from tpu_sync.api.torch import weight_synchronizer as ws_mod

        _ws_lib = ws_mod
        _RAIDEN_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - environment dependent
        _RAIDEN_IMPORT_ERROR = exc
    return _ws_lib


def is_parallel_h2h_enabled() -> bool:
    """Return whether explicit two-phase H2H + H2D sync is enabled."""
    return bool(envs.WEIGHT_SYNC_PARALLEL_H2H)


def local_ip() -> str:
    """Resolve the local advertise IP address for Raiden endpoints."""
    if envs.RAIDEN_BIND_IP:
        return envs.RAIDEN_BIND_IP
    for family, probe in (
        (socket.AF_INET, ("8.8.8.8", 80)),
        (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
    ):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect(probe)
                ip = sock.getsockname()[0]
            return f"[{ip}]" if ":" in ip else ip
        except OSError:
            continue
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _resolve_data_nic_ips(default_ip: str) -> list[str]:
    """Resolve IPv4 addresses for interfaces listed in TPU_RAIDEN_DATA_NICS."""
    if not envs.TPU_RAIDEN_DATA_NICS:
        return [default_ip]
    ips: list[str] = []
    try:
        import fcntl
        import struct

        for nic in envs.TPU_RAIDEN_DATA_NICS.split(","):
            nic = nic.strip()
            if not nic:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    info = fcntl.ioctl(
                        sock.fileno(),
                        0x8915,  # SIOCGIFADDR
                        struct.pack("256s", nic[:15].encode("utf-8")),
                    )
                ips.append(socket.inet_ntoa(info[20:24]))
            except OSError:
                continue
    except ImportError:
        pass
    return ips or [default_ip]


def _routable_endpoint(addr: str, advertise_ip: str) -> str:
    """Replace wildcard or loopback host in ``addr`` with ``advertise_ip``."""
    host, sep, port = addr.rpartition(":")
    if sep and host in _WILDCARD_HOSTS:
        return f"{advertise_ip}:{port}"
    return addr


@dataclass
class BoundVariable:
    """Destination parameter bound to Raiden plus runtime layout metadata."""

    name: str
    tensor: torch.Tensor
    shard_dim: int | None
    num_shards: int
    shard_domain: str
    global_shape: tuple[int, ...]
    flipped: bool = False
    owner_cls: str = ""

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.tensor.shape)

    def layout_dict(self) -> dict[str, Any]:
        return {
            "local_shape": list(self.local_shape),
            "global_shape": list(self.global_shape),
            "dtype": str(self.tensor.dtype).removeprefix("torch."),
            "shard_dim": self.shard_dim,
            "num_shards": self.num_shards,
            "shard_domain": self.shard_domain,
            "flipped": self.flipped,
        }


def canonical_param_name(name: str) -> str:
    """Normalize a vLLM parameter path to the canonical dotted sync key."""
    for old, new in _NAME_REWRITES:
        name = name.replace(old, new)
    segments: list[str] = []
    for raw in name.split("."):
        comp = raw.strip("'\"")
        if not comp:
            continue
        folded = _FOLDED_INDEX_RE.match(comp)
        segments.extend(folded.groups() if folded else (comp,))
    while segments and segments[0] in _WRAPPER_ROOTS:
        segments.pop(0)
    if segments and segments[-1] == "value":
        segments.pop()
    return ".".join(segments)


def _language_model_and_skipped_roots(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, set[str]]:
    get_lm = getattr(model, "get_language_model", None)
    lm_model = get_lm() if callable(get_lm) else model
    if not isinstance(lm_model, torch.nn.Module):
        lm_model = model

    skipped_roots: set[str] = set(_SKIP_PREFIXES)
    for root in (model, lm_model):
        for tower_name in getattr(root, "_tower_model_names", ()) or ():
            skipped_roots.add(str(tower_name))
        for sub_name, sub_mod in root.named_modules():
            if sub_name and getattr(sub_mod, "_is_tower_model", False):
                skipped_roots.add(str(sub_name))
    return lm_model, skipped_roots


def _is_skipped_path(name: str, skipped_roots: set[str]) -> bool:
    cur = name
    while True:
        if any(cur == r or cur.startswith(f"{r}.") for r in skipped_roots):
            return True
        head, sep, rest = cur.partition(".")
        if not sep or head not in _WRAPPER_ROOTS:
            return False
        cur = rest


def _is_tpu_moe_flipped(module: torch.nn.Module) -> bool:
    explicit = getattr(module, WEIGHT_FLIPPED_ATTR, None)
    if explicit is not None:
        return bool(explicit)
    qmethod = getattr(module, "quant_method", None)
    return bool(
        hasattr(module, "_tpu_activation")
        or (
            qmethod is not None and type(qmethod).__module__.startswith("vllm_torchtpu")
        )
    )


def classify_parameter(
    model: torch.nn.Module,
    param_name: str,
    param: torch.Tensor,
    tp_size: int,
) -> BoundVariable:
    """Inspect owning ``nn.Module`` to derive sharding and storage layout."""
    mod_path, _, attr = param_name.rpartition(".")
    module = model.get_submodule(mod_path) if mod_path else model
    owner_cls = type(module).__name__
    cname = canonical_param_name(param_name)

    def _build(
        shard_dim: int | None,
        num_shards: int,
        shard_domain: str,
        *,
        flipped: bool = False,
    ) -> BoundVariable:
        if shard_dim is None or num_shards <= 1:
            shard_dim, num_shards, shard_domain = None, 1, "replicated"
        gshape = [int(d) for d in param.shape]
        if shard_dim is not None:
            gshape[shard_dim] *= num_shards
        return BoundVariable(
            name=cname,
            tensor=param,
            shard_dim=shard_dim,
            num_shards=num_shards,
            shard_domain=shard_domain,
            global_shape=tuple(gshape),
            flipped=flipped,
            owner_cls=owner_cls,
        )

    if isinstance(module, (VocabParallelEmbedding, ParallelLMHead)):
        mod_tp = int(getattr(module, "tp_size", tp_size))
        domain = "ep" if bool(getattr(module, DP_VOCAB_ATTR, False)) else "tp"
        if mod_tp > 1:
            org = int(getattr(module, "org_vocab_size", param.shape[0] * mod_tp))
            if param.shape[0] * mod_tp != org:
                raise ValueError(
                    f"{param_name}: padded vocab shard {tuple(param.shape)} * "
                    f"{domain}={mod_tp} != org_vocab_size={org}."
                )
        return _build(0 if mod_tp > 1 else None, mod_tp, domain)

    if isinstance(module, RoutedExperts):
        par = module.moe_config.moe_parallel_config
        moe_flipped = _is_tpu_moe_flipped(module)
        if par.use_ep:
            ep = int(par.ep_size)
            return _build(0 if ep > 1 else None, ep, "ep", flipped=moe_flipped)
        moe_tp = int(par.tp_size)
        if attr == "w13_weight":
            sdim: int | None = 2 if moe_flipped else 1
        elif attr == "w2_weight":
            sdim = 1 if moe_flipped else 2
        else:
            raise ValueError(f"{param_name}: unsupported TP-MoE parameter {attr!r}")
        return _build(sdim, moe_tp, "tp", flipped=moe_flipped)

    if isinstance(module, LinearBase):
        flipped = bool(getattr(module, WEIGHT_FLIPPED_ATTR, False))
        if isinstance(module, ReplicatedLinear) or bool(
            getattr(module, "disable_tp", False)
        ):
            return _build(None, 1, "replicated", flipped=flipped)
        mod_tp = int(getattr(module, "tp_size", tp_size))
        if param.dim() == 3:
            # GDN depthwise conv1d stored as [local_channels, 1, kernel_size].
            return _build(0, mod_tp, "tp")
        if isinstance(module, RowParallelLinear):
            sdim = (0 if flipped else 1) if attr == "weight" else None
            return _build(sdim, mod_tp, "tp", flipped=flipped)
        sdim = 1 if (flipped and attr == "weight") else 0
        return _build(sdim, mod_tp, "tp", flipped=flipped)

    if (isinstance(module, MambaBase) or attr in _HEAD_SHARDED_ATTRS) and (
        attr in _HEAD_SHARDED_ATTRS or getattr(param, "weight_loader", None) is not None
    ):
        mod_tp = int(getattr(module, "tp_size", tp_size))
        return _build(0 if mod_tp > 1 else None, mod_tp, "tp")

    return _build(None, 1, "replicated")


def collect_variables(
    model: torch.nn.Module,
    tp_size: int,
) -> list[BoundVariable]:
    """Enumerate and canonicalize bindable language-model parameters."""
    lm_model, skipped_roots = _language_model_and_skipped_roots(model)
    out: list[BoundVariable] = []
    seen_ptrs: dict[int, str] = {}
    seen_names: dict[str, str] = {}

    for name, param in lm_model.named_parameters():
        if _is_skipped_path(name, skipped_roots):
            continue
        tensor = param.data
        ptr = tensor.data_ptr()
        var = classify_parameter(lm_model, name, tensor, tp_size)
        if not var.name:
            raise ValueError(f"Parameter {name!r} canonicalized to an empty key.")
        if ptr in seen_ptrs:
            logger.info_once(
                "[raiden_sync] Skipping tied parameter %s sharing buffer with %s.",
                var.name,
                seen_ptrs[ptr],
            )
            continue
        if var.name in seen_names:
            raise ValueError(
                f"Parameters {seen_names[var.name]!r} and {name!r} both "
                f"canonicalize to {var.name!r}."
            )
        seen_ptrs[ptr] = var.name
        seen_names[var.name] = name
        out.append(var)

    if not out:
        raise RuntimeError("No bindable language-model parameters found on model.")

    # Raiden pairs by name for slice planning, while C++ indexes destination
    # buffers by `src_var.layer_idx`. Sorting post-canonicalization by dotted
    # path tuple matches JAX pytree key traversal on the trainer side.
    out.sort(key=lambda v: tuple(v.name.split(".")))
    return out


def _check_bindable(var: BoundVariable) -> None:
    t = var.tensor
    if t.device.type != "tpu":
        raise ValueError(f"{var.name}: expected TPU tensor, got device={t.device}")
    if not t.is_contiguous() or t.storage_offset() != 0:
        raise ValueError(
            f"{var.name}: expected contiguous root view (offset={t.storage_offset()})"
        )
    if t.dim() < 1 or t.shape[0] == 0:
        raise ValueError(f"{var.name}: empty or 0-D tensor {tuple(t.shape)}")


class RaidenWorkerSync:
    """Per-worker live parameter binding and metadata publisher for Raiden."""

    def __init__(
        self,
        job_name: str,
        *,
        worker_index: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        dp_rank: int = 0,
        dp_size: int = 1,
        parallelism: int = 4,
        bind_ip: str | None = None,
        auto_h2d: bool | None = None,
    ) -> None:
        self.job_name = job_name
        self.worker_index = int(worker_index)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        default_ip = bind_ip or local_ip()
        nic_ips = _resolve_data_nic_ips(default_ip)
        self.ip = (
            bind_ip
            or nic_ips[(self.dp_rank * self.tp_size + self.tp_rank) % len(nic_ips)]
        )
        env_par = envs.RAIDEN_PARALLELISM
        self._parallelism = int(env_par) if env_par is not None else int(parallelism)
        self._auto_h2d = (
            (not is_parallel_h2h_enabled()) if auto_h2d is None else bool(auto_h2d)
        )
        self._effective_auto_h2d = self._auto_h2d
        self.variables: list[BoundVariable] = []
        self._bound_ptrs: list[int] = []
        self._sync: Any = None

    @property
    def bound(self) -> bool:
        return self._sync is not None and bool(self.variables)

    def bind(self, model: torch.nn.Module, *, auto_h2d: bool | None = None) -> None:
        """Bind model parameters on this TPU worker to a 1-chip synchronizer."""
        ws_lib = _get_ws_lib()
        if ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable: {_RAIDEN_IMPORT_ERROR}"
            )
        if auto_h2d is not None:
            requested = bool(auto_h2d)
            if self._sync is not None and requested != self._effective_auto_h2d:
                logger.warning(
                    "[raiden_sync] %s: ignoring auto_h2d=%s on rebind "
                    "(synchronizer already created with auto_h2d=%s).",
                    self.job_name,
                    requested,
                    self._effective_auto_h2d,
                )
            else:
                self._auto_h2d = requested

        variables = collect_variables(model, self.tp_size)
        for v in variables:
            _check_bindable(v)
        tensors = [v.tensor for v in variables]

        try:
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(tensors)
        except Exception:  # pragma: no cover
            torch.accelerator.synchronize()

        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        ptrs = [t.data_ptr() for t in tensors]
        if self._sync is None:
            logger.info(
                "[raiden_sync] %s[dp=%d/%d,tp=%d/%d]: binding %d weight tensors "
                "(%.2f GiB local).",
                self.job_name,
                self.dp_rank,
                self.dp_size,
                self.tp_rank,
                self.tp_size,
                len(tensors),
                nbytes / 2**30,
            )
            self._sync = ws_lib.WeightSynchronizer(
                [[t] for t in tensors],
                local_port=0,
                parallelism=self._parallelism,
                listener_port=0,
                bind_ip=envs.RAIDEN_BIND_IP or None,
                unsafe_skip_buffer_lock=True,
                auto_h2d=self._auto_h2d,
                global_shard_indices=[0],
            )
            self._effective_auto_h2d = self._auto_h2d
        elif ptrs != self._bound_ptrs or len(tensors) != len(self.variables):
            logger.info(
                "[raiden_sync] %s[dp=%d/%d,tp=%d/%d]: parameter buffers moved; "
                "re-binding %d tensors (%.2f GiB local).",
                self.job_name,
                self.dp_rank,
                self.dp_size,
                self.tp_rank,
                self.tp_size,
                len(tensors),
                nbytes / 2**30,
            )
            self._sync.bind_weights([[t] for t in tensors])
        self.variables = variables
        self._bound_ptrs = ptrs

    def _require(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}.")
        return self._sync

    def h2d(self, uuid: int | None = None) -> None:
        """Wait for transfer completion and flush device H2D copies."""
        sync = self._require("h2d()")
        t0 = time.monotonic()
        if not self._effective_auto_h2d:
            if uuid is None or int(uuid) <= 0:
                raise ValueError(
                    f"{self.job_name}: h2d() with auto_h2d=False requires "
                    f"uuid > 0, got {uuid!r}."
                )
            sync.wait_for_transfer_completion(int(uuid))
            sync.h2d()
        else:
            sync.wait_for_transfer_completion(uuid)
        torch.accelerator.synchronize()
        logger.info(
            "[raiden_sync] %s: h2d complete in %.2fs (uuid=%s).",
            self.job_name,
            time.monotonic() - t0,
            uuid,
        )

    def metrics(self) -> dict[str, Any]:
        return dict(self._sync.get_metrics()) if self._sync is not None else {}

    def checksums(self, sample: int = 3) -> dict[str, float | int]:
        """Compute per-tensor float32 L1 sums in a single device-to-host sync."""
        if not self.variables:
            return {
                "__grand_total__": 0.0,
                "__tensor_count__": 0,
                "__element_count__": 0,
            }
        per_tensor = (
            torch.stack(
                [
                    v.tensor.detach().abs().sum(dtype=torch.float32)
                    for v in self.variables
                ]
            )
            .cpu()
            .tolist()
        )
        out: dict[str, float | int] = {
            v.name: float(val)
            for v, val in zip(self.variables[:sample], per_tensor[:sample])
        }
        out["__grand_total__"] = float(math.fsum(per_tensor))
        out["__tensor_count__"] = len(self.variables)
        out["__element_count__"] = int(sum(v.tensor.numel() for v in self.variables))
        return out

    def describe(self) -> str:
        """Format bound variable shapes and sharding for debug diagnostics."""
        lines = [
            f"{self.job_name}[dp={self.dp_rank}/{self.dp_size},"
            f"tp={self.tp_rank}/{self.tp_size}] {len(self.variables)} variables"
        ]
        for v in self.variables:
            lines.append(
                f"  {v.name}: local={v.local_shape} global={v.global_shape} "
                f"domain={v.shard_domain} dim={v.shard_dim} "
                f"flipped={v.flipped} dtype={v.tensor.dtype} ({v.owner_cls})"
            )
        return "\n".join(lines)

    def variable_dicts(self) -> list[dict[str, Any]]:
        """Per-tensor registration metadata matching RaidenHandler invariants."""
        out: list[dict[str, Any]] = []
        ep_size_phys = self.dp_size * self.tp_size
        ep_rank = self.dp_rank * self.tp_size + self.tp_rank
        ep_axis = "dp,tp" if self.dp_size > 1 else "tp"
        for idx, v in enumerate(self.variables):
            nd = len(v.global_shape)
            mesh_shape = [1] * nd
            spec = [""] * nd
            gsi = [0]
            if v.shard_dim is not None:
                mesh_shape[v.shard_dim] = v.num_shards
                if v.shard_domain == "ep":
                    if v.num_shards != ep_size_phys:
                        raise NotImplementedError(
                            f"{v.name}: EP shard count {v.num_shards} does not "
                            f"match worker EP mesh size {ep_size_phys} "
                            f"(dp={self.dp_size} * tp={self.tp_size})."
                        )
                    spec[v.shard_dim] = ep_axis
                    gsi = [ep_rank]
                else:
                    if v.num_shards != self.tp_size:
                        raise NotImplementedError(
                            f"{v.name}: TP shard count {v.num_shards} does not "
                            f"match worker tp_size={self.tp_size}."
                        )
                    spec[v.shard_dim] = "tp"
                    gsi = [self.tp_rank]
            out.append(
                {
                    "name": v.name,
                    "shape": list(v.global_shape),
                    "mesh_shape": mesh_shape,
                    "layout": list(reversed(range(nd))),
                    "item_size": v.tensor.element_size(),
                    "layer_idx": idx,
                    "sharding_spec": spec,
                    "global_shard_indices": gsi,
                }
            )
        return out

    def metadata_dict(self) -> dict[str, Any]:
        """Wire-safe registration metadata shaped for WorkUnitMetadata.from_dict."""
        sync = self._require("metadata_dict()")
        num_shards = int(sync.num_shards)
        shards_list = [""] * num_shards
        try:
            for ep in sync.get_local_endpoints():
                addr = _routable_endpoint(str(ep.get("endpoint", "")), self.ip)
                for s in ep.get("shards", []):
                    if 0 <= s < num_shards:
                        shards_list[s] = addr
        except Exception as exc:  # pragma: no cover
            logger.debug("[raiden_sync] get_local_endpoints fallback: %s", exc)
        if not all(shards_list):
            shards_list = [s or f"{self.ip}:{sync.local_port}" for s in shards_list]

        variables = self.variable_dicts()
        for var in variables:
            if len(var["global_shard_indices"]) != len(shards_list):
                raise RuntimeError(
                    f"{self.job_name}: variable {var['name']!r} publishes "
                    f"{len(var['global_shard_indices'])} global_shard_indices "
                    f"for {len(shards_list)} local shard(s)."
                )

        replica_rank = (
            self.worker_index * self.dp_size * self.tp_size
            + self.dp_rank * self.tp_size
            + self.tp_rank
        )
        meta: dict[str, Any] = {
            "unit": {"job_name": self.job_name, "job_replica_id": str(replica_rank)},
            "shards": shards_list,
            "control_plane_rpc_address": (
                f"{self.ip}:{sync.listener_port}" if sync.listener_port else ""
            ),
            "mesh_shape": (
                [self.dp_size, self.tp_size] if self.dp_size > 1 else [self.tp_size]
            ),
            "mesh_axes": ["dp", "tp"] if self.dp_size > 1 else ["tp"],
            "host_subgrid": None,
            "variables": variables,
            "auto_h2d": bool(self._effective_auto_h2d),
        }
        if self.dp_rank == 0 and self.tp_rank == 0:
            meta["param_layouts"] = {v.name: v.layout_dict() for v in self.variables}
        return meta
