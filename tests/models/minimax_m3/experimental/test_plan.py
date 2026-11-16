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
"""MiniMax M3 model plan: profile names resolve and import, the default plan
matches its snapshot, and experimental/ reads no global switches."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_torchtpu.models.minimax_m3 import experimental
from vllm_torchtpu.models.minimax_m3.experimental import plan, profiles, variants

_EXPERIMENTAL = Path(experimental.__file__).parent
_SNAPSHOT = _EXPERIMENTAL / "validation" / "plan_snapshot.json"


def _config(
    *,
    tp=8,
    dp=1,
    pcp=1,
    ep=True,
    spec=True,
    external_lb=False,
    additional=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            enable_expert_parallel=ep,
            data_parallel_external_lb=external_lb,
        ),
        speculative_config=SimpleNamespace(method="dspark") if spec else None,
        additional_config=additional if additional is not None else {},
    )


def test_every_profile_pins_every_slot_with_known_variants():
    assert profiles.DEFAULT_PROFILE in profiles.PROFILES
    for name, slots in profiles.PROFILES.items():
        assert set(slots) == set(variants.VARIANTS), name
        for slot, choices in slots.items():
            assert choices, (name, slot)
            for choice in choices:
                assert choice in variants.VARIANTS[slot], (name, slot, choice)


@pytest.mark.parametrize(
    "slot,name",
    [(s, n) for s, table in variants.VARIANTS.items() for n in table],
)
def test_every_variant_resolves_and_imports(slot, name):
    variant = variants.VARIANTS[slot][name]
    impl = variant.load()
    if isinstance(variant.impl, str):
        assert impl is not None and callable(impl)


def test_registered_classes_match_the_catalog():
    assert variants.VARIANTS["model"]["msa_sparse"].impl == experimental._MODEL
    assert variants.VARIANTS["draft"]["dspark"].impl == experimental._DSPARK_MODEL


def test_default_plan_matches_snapshot():
    """A change of the default selection shows as a diff of the snapshot."""
    snapshot = json.loads(_SNAPSHOT.read_text())
    got = {
        case: plan.resolve_plan(_config(**entry["config"]))
        for case, entry in snapshot.items()
    }
    want = {case: entry["plan"] for case, entry in snapshot.items()}
    assert got == want, json.dumps(got, indent=1, sort_keys=True)


def test_profile_selection():
    assert plan.profile_name(_config()) == profiles.DEFAULT_PROFILE
    assert plan.profile_name(_config(additional={"model_plan": {}})) == (
        profiles.DEFAULT_PROFILE
    )
    cfg = _config(dp=2, tp=4, additional={"model_plan": {"profile": "tp4-dp2"}})
    assert plan.selected(cfg, "dp_routing") is True
    assert plan.prefix_affinity_policy(cfg)
    assert not plan.prefix_affinity_policy(_config(dp=2, tp=4))
    for bad in ({"profile": "nope"}, {"profle": "agentx"}, "agentx"):
        with pytest.raises(ValueError):
            plan.profile_name(_config(additional={"model_plan": bad}))


_GLOBAL_READS = re.compile(
    r"os\.environ|os\.getenv|\bgetenv\(|import envs\b|\benvs\.[A-Z_]"
    r"|vllm\.envs|additional_config"
)


def test_experimental_reads_no_global_switches():
    """Only plan.py reads additional_config (model_plan); nothing reads env."""
    offenders = []
    for path in sorted(_EXPERIMENTAL.rglob("*.py")):
        if path.name == "plan.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if _GLOBAL_READS.search(code):
                offenders.append(f"{path.relative_to(_EXPERIMENTAL)}:{lineno}")
    assert not offenders, offenders
