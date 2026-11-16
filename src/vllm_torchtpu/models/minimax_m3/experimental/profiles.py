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
"""Named MiniMax M3 recipes: profile -> {slot: variants in order of preference}.

Every profile pins every slot of `variants.VARIANTS`. `DEFAULT_PROFILE` is
the proven-best served recipe (the AgentX curve: TP8, EP, DCP 2 / 4 / 8,
DSpark); `vllm serve` uses it with no extra flags. Pick another one with
``--additional-config '{"model_plan": {"profile": "<name>"}}'``.
"""

DEFAULT_PROFILE = "agentx"

PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    # Served best (v2d / v3 FULLs): Pallas TP all-reduce up to 512 KiB views,
    # fused spec verify; one API server routes as upstream.
    "agentx": {
        "model": ("msa_sparse",),
        "draft": ("dspark",),
        "tp_all_reduce": ("pallas_512k", "xla"),
        "spec_verify": ("fused", "unfused"),
        "dp_routing": ("load_balance",),
    },
    # Data-parallel engines (e.g. TP4 DP2): route requests by prefix
    # affinity (needs --api-server-count=1). TP4 DP2 + affinity led c48/c64
    # before TP8 + DCP (c64 FAST +47% over TP4 DP2 without it); TP8 + DCP 2
    # beats it at c16 (+13%), so it is not the default.
    "tp4-dp2": {
        "model": ("msa_sparse",),
        "draft": ("dspark",),
        "tp_all_reduce": ("pallas_512k", "xla"),
        "spec_verify": ("fused", "unfused"),
        "dp_routing": ("prefix_affinity", "load_balance"),
    },
}
