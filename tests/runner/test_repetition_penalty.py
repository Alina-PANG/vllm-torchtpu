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
"""``repetition_penalty`` on the TPU sampler: request tracking, packed bitmask
scaling math, sync decode integration, and TPU compilation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from runner_input_test_utils import plain_step_runner, run_sample_tokens
from vllm import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_torchtpu.runner import greedy_top_tokens, min_tokens
from vllm_torchtpu.runner import repetition_penalty as rep_pen
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

NEG_INF = float("-inf")


def _unpack_bits(bitmask: np.ndarray, vocab_size: int) -> list[int]:
    shifts = np.arange(32, dtype=np.uint32)
    bits = ((bitmask.view(np.uint32)[:, None] >> shifts[None, :]) & 1) != 0
    return np.where(bits.reshape(-1)[:vocab_size])[0].tolist()


def _pack_bool_mask(seen_bool: np.ndarray) -> np.ndarray:
    num_rows, vocab_size = seen_bool.shape
    num_words = cdiv(vocab_size, 32)
    out = np.zeros((num_rows, num_words), dtype=np.int32)
    for r in range(num_rows):
        for tok in np.where(seen_bool[r])[0]:
            out[r, int(tok) >> 5] |= rep_pen._BIT_MASKS[int(tok) & 31]
    return out


@pytest.fixture
def eager():
    with torch.compiler.set_stance("force_eager"):
        yield


def _batch(rows: dict[str, tuple[float, list[int]]], order: list[str]):
    """An ``InputBatch`` stand-in: ``order`` gives the batch rows and ``rows``
    maps a request id to its ``(repetition_penalty, output_token_ids)``."""
    return SimpleNamespace(
        req_ids=list(order),
        req_id_to_index={r: i for i, r in enumerate(order)},
        repetition_penalties_reqs={r for r in order if rows[r][0] != 1.0},
        repetition_penalties_cpu=np.array([rows[r][0] for r in order], np.float32),
        req_output_token_ids=[rows[r][1] for r in order],
    )


def _tracker_in_mode(vocab_size: int, async_scheduling: bool, spec):
    return rep_pen.RepetitionPenaltyTracker(
        vocab_size,
        8,
        async_scheduling=async_scheduling,
        speculative_decoding=spec is not None,
    )


def _tracker(vocab_size: int, max_num_reqs: int = 8):
    return rep_pen.RepetitionPenaltyTracker(
        vocab_size, max_num_reqs, async_scheduling=False, speculative_decoding=False
    )


class TestSeenTokenBitmask:
    def test_incremental_update_rollback_and_out_of_range_ids(self):
        seen = rep_pen.SeenTokenBitmask(vocab_size=64)
        # Includes -1 (MM placeholder), 31 (sign bit of word 0), and 151645 (OOB).
        prompt = [-1, 2, 5, 31, 151645]
        output: list[int] = []

        # Prefill: only in-range prompt ids [2, 5, 31] marked.
        m0 = seen.update(prompt, output)
        assert _unpack_bits(m0, 64) == [2, 5, 31]

        # Incremental decode update (+ out-of-range ignored + bit 63 sign bit).
        output += [6, 63, 999]
        m1 = seen.update(prompt, output)
        assert _unpack_bits(m1, 64) == [2, 5, 6, 31, 63]

        # Additional output tokens update in place without reallocating.
        output += [1] * 10
        m2 = seen.update(prompt, output)
        assert m2 is m1
        assert _unpack_bits(m2, 64) == [1, 2, 5, 6, 31, 63]

        # Shrunk output (rollback) zeroes and rebuilds in place from prompt + output.
        del output[1:]
        m3 = seen.update(prompt, output)
        assert m3 is m1
        assert _unpack_bits(m3, 64) == [2, 5, 6, 31]

    def test_new_prompt_or_output_list_rebuilds(self):
        """A streaming-input continuation, or a finished id sent again, gives
        the request new lists; the old bits must not leak into the new ones."""
        seen = rep_pen.SeenTokenBitmask(vocab_size=16)
        output = [7]
        assert _unpack_bits(seen.update([2], output), 16) == [2, 7]
        # Same output list, new prompt (streaming continuation keeps the list).
        output.clear()
        assert _unpack_bits(seen.update([2, 9], output), 16) == [2, 9]
        # New output list object with the same length.
        assert _unpack_bits(seen.update([2, 9], []), 16) == [2, 9]
        assert _unpack_bits(seen.update(None, [4]), 16) == [4]

    def test_numpy_prompt_ids(self):
        seen = rep_pen.SeenTokenBitmask(vocab_size=32)
        prompt = np.array([1, 3, 31], dtype=np.int32)
        assert _unpack_bits(seen.update(prompt, []), 32) == [1, 3, 31]


class TestTracker:
    def test_reads_rows_penalties_and_outputs_from_the_batch(self):
        tracker = _tracker(vocab_size=64)
        requests = {
            "a": SimpleNamespace(prompt_token_ids=[1]),
            "b": SimpleNamespace(prompt_token_ids=[2]),
        }
        rows = {"a": (1.1, [3]), "b": (1.0, [4])}
        # Row 0 ("b") has no penalty; row 1 ("a") does.
        inputs = tracker.build_plain_inputs(
            _batch(rows, ["b", "a"]), requests, 0, 2, num_rows=4
        )
        assert inputs is not None
        bitmask, penalties = inputs
        assert _unpack_bits(bitmask[1], 64) == [1, 3]
        assert _unpack_bits(bitmask[0], 64) == []
        assert penalties[:, 0].tolist() == pytest.approx([1.0, 1.1, 1.0, 1.0])

        # The next step moves "a" to row 0: row 1 is cleared in the reused buffer.
        inputs2 = tracker.build_plain_inputs(
            _batch(rows, ["a", "b"]), requests, 0, 2, num_rows=4
        )
        assert inputs2 is not None
        bitmask2, penalties2 = inputs2
        assert np.shares_memory(bitmask2, bitmask)
        assert np.shares_memory(penalties2, penalties)
        assert _unpack_bits(bitmask2[0], 64) == [1, 3]
        assert _unpack_bits(bitmask2[1], 64) == []
        assert penalties2[:, 0].tolist() == pytest.approx([1.1, 1.0, 1.0, 1.0])

        # A chunk without a penalized row stages nothing.
        assert (
            tracker.build_plain_inputs(
                _batch(rows, ["a", "b"]), requests, 1, 2, num_rows=4
            )
            is None
        )

    def test_prune_drops_requests_that_left_the_penalized_set(self):
        tracker = _tracker(vocab_size=8)
        requests = {r: SimpleNamespace(prompt_token_ids=[1]) for r in "ab"}
        rows = {"a": (1.1, []), "b": (1.2, [])}
        tracker.build_plain_inputs(_batch(rows, ["a", "b"]), requests, 0, 2, num_rows=2)
        assert sorted(tracker._seen) == ["a", "b"]
        assert tracker.begin_step(_batch(rows, ["b"]))
        assert sorted(tracker._seen) == ["b"]


@pytest.mark.usefixtures("eager")
class TestScalingMath:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("vocab_size", [64, 70])
    def test_matches_upstream_vllm_reference(self, dtype, vocab_size):
        from vllm._custom_ops import apply_repetition_penalties

        gen = torch.Generator().manual_seed(42)
        num_rows = 6
        logits = torch.randn((num_rows, vocab_size), generator=gen).to(dtype)
        logits[0, 0] = 0.0
        logits[1, 5] = NEG_INF

        prompt_mask = torch.zeros((num_rows, vocab_size), dtype=torch.bool)
        output_mask = torch.zeros((num_rows, vocab_size), dtype=torch.bool)
        prompt_mask[0, [0, 3, 12, 31]] = True
        output_mask[0, [7, 12]] = True
        prompt_mask[1, [5, 9]] = True
        output_mask[3, [1, 2, 63]] = True
        # Row 2 has penalty == 1.0 (untracked), row 4 has penalty < 1.0.
        penalties_1d = torch.tensor(
            [1.05, 1.5, 1.0, 2.0, 0.8, 1.0], dtype=torch.float32
        )
        output_mask[4, [10, 20]] = True

        ref = logits.clone()
        apply_repetition_penalties(ref, prompt_mask, output_mask, penalties_1d)

        seen_bitmask_np = _pack_bool_mask((prompt_mask | output_mask).numpy())
        pen_np = penalties_1d.view(-1, 1).numpy()
        got = rep_pen.apply_penalties_table(logits, seen_bitmask_np, pen_np)

        assert got.dtype == dtype
        assert torch.isneginf(got[1, 5])
        torch.testing.assert_close(
            got[~torch.isneginf(ref)],
            ref[~torch.isneginf(ref)],
            rtol=1e-2 if dtype == torch.bfloat16 else 1e-6,
            atol=1e-2 if dtype == torch.bfloat16 else 1e-6,
        )


def _penalized_runner(logits: torch.Tensor, penalty: float = 1.5):
    """`plain_step_runner` state where request `r0` has prompt `[4]` and
    `repetition_penalty` `penalty`."""
    runner = plain_step_runner(logits)
    runner.requests["r0"].prompt_token_ids = [4]
    runner.input_batch.repetition_penalties_reqs.add("r0")
    runner.input_batch.repetition_penalties_cpu[0] = penalty
    return runner


@pytest.mark.usefixtures("eager")
class TestRunnerIntegration:
    def test_plain_step_penalizes_seen_tokens_and_preserves_raw_logprobs(self):
        # Token 4 (in prompt) has logit 3.0; token 2 (unseen) has logit 2.5.
        # With penalty 1.5, token 4 drops to 2.0, so greedy picks token 2,
        # while raw logprobs still read 3.0 on token 4.
        logits = torch.zeros(1, 8)
        logits[0, 4], logits[0, 2] = 3.0, 2.5
        runner = _penalized_runner(logits)
        runner.input_batch.max_num_logprobs = 0
        seen_logprob_logits: list[torch.Tensor] = []

        def gather_logprobs(x, tokens):
            seen_logprob_logits.append(x.clone())
            return LogprobsTensors(
                tokens.view(-1, 1), torch.zeros(1, 1), torch.zeros(1, dtype=torch.int64)
            )

        runner.gather_logprobs = gather_logprobs
        out = run_sample_tokens(runner)
        assert out.sampled_token_ids == [[2]]
        assert seen_logprob_logits[0][0, 4].item() == pytest.approx(3.0)

    def test_batch_reorder_and_top_tokens_gate(self):
        batch = InputBatch(
            max_num_reqs=4,
            max_model_len=64,
            max_num_batched_tokens=16,
            device=torch.device("cpu"),
            vocab_size=16,
            block_sizes=[16],
            max_num_blocks_per_req=[4],
            logitsprocs=LogitsProcessors(),
            logitsprocs_need_output_token_ids=True,
        )
        a = CachedRequestState(
            req_id="a",
            prompt_token_ids=[3],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0, repetition_penalty=2.0),
            generator=None,
            block_ids=([0],),
            num_computed_tokens=1,
            output_token_ids=[5],
        )
        b = CachedRequestState(
            req_id="b",
            prompt_token_ids=[3],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            generator=None,
            block_ids=([1],),
            num_computed_tokens=1,
            output_token_ids=[5],
        )
        batch.add_request(a)
        batch.add_request(b)
        batch.refresh_metadata()
        batch.swap_states(0, 1)  # "a" moves to row 1

        runner = object.__new__(TPUModelRunner)
        runner.input_batch = batch
        runner.requests = {"a": a, "b": b}
        runner._pre_async_results = None
        runner.scheduler_config = SimpleNamespace(async_scheduling=False)
        runner.speculative_config = None
        runner._greedy_head_owner = object()
        runner._min_tokens = min_tokens.MinTokensTracker()
        runner._repetition_penalty = rep_pen.RepetitionPenaltyTracker(
            vocab_size=16,
            max_num_reqs=4,
            async_scheduling=False,
            speculative_decoding=False,
        )

        assert batch.repetition_penalties_reqs == {"a"}
        assert not runner._greedy_top_tokens_this_step()
        assert runner._repetition_penalty.begin_step(batch)
        logits = torch.full((4, 16), 2.0)
        out = runner._repetition_penalty.apply(logits, batch, runner.requests, 0, 2)
        # Row 0 ("b") untouched; row 1 ("a") scaled at tokens 3 and 5.
        assert torch.equal(out[0], logits[0])
        assert out[1, 3].item() == pytest.approx(1.0)
        assert out[1, 5].item() == pytest.approx(1.0)
        assert out[1, 0].item() == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("async_scheduling", "spec"), [(True, None), (False, object())]
    )
    def test_unsupported_modes_skip_penalty_with_warning(self, async_scheduling, spec):
        """An unsupported mode logs warning_once, applies nothing and keeps the
        greedy top-tokens fast path open."""
        runner = _penalized_runner(torch.zeros(1, 8))
        runner._repetition_penalty = _tracker_in_mode(8, async_scheduling, spec)
        with patch.object(rep_pen, "logger") as log:
            assert not runner._repetition_penalty.begin_step(runner.input_batch)
        log.warning_once.assert_called_once()
        assert "async_scheduling=False" in log.warning_once.call_args.args[0]
        runner.input_batch.all_greedy = True
        runner._greedy_head_owner = object()
        with patch.object(greedy_top_tokens, "step_eligible", return_value=True):
            assert TPUModelRunner._greedy_top_tokens_this_step(runner)

    @pytest.mark.parametrize(
        ("async_scheduling", "spec", "expected"),
        [
            (
                False,
                None,
                [(8, 12, torch.bfloat16, (8, 1)), (16, 12, torch.bfloat16, (16, 1))],
            ),
            (True, None, []),
            (False, object(), []),
        ],
    )
    def test_precompile_covers_supported_buckets_only(
        self, async_scheduling, spec, expected
    ):
        runner = object.__new__(TPUModelRunner)
        runner.num_reqs_paddings = [8, 16]
        runner._repetition_penalty = rep_pen.RepetitionPenaltyTracker(
            vocab_size=12,
            max_num_reqs=16,
            async_scheduling=async_scheduling,
            speculative_decoding=spec is not None,
        )
        runner.device = torch.device("cpu")
        runner._dummy_logits = lambda n: torch.zeros(n, 12, dtype=torch.bfloat16)
        runner._precompile_timed = MagicMock()
        apply_calls: list[tuple[int, int, torch.dtype, tuple[int, int]]] = []
        orig_apply = rep_pen.apply_repetition_penalty

        def spy_apply(logits, seen_bitmask, penalties):
            assert seen_bitmask.dtype == torch.int32
            apply_calls.append(
                (
                    logits.shape[0],
                    logits.shape[1],
                    logits.dtype,
                    tuple(seen_bitmask.shape),
                )
            )
            return orig_apply(logits, seen_bitmask, penalties)

        with patch.object(rep_pen, "apply_repetition_penalty", side_effect=spy_apply):
            runner._precompile_repetition_penalty()

        assert apply_calls == expected


@pytest.mark.tpu_test
def test_compiled_repetition_penalty_on_tpu():
    import torch_tpu  # noqa: F401

    device = torch.device("tpu")
    gen = torch.Generator().manual_seed(7)
    logits_cpu = torch.randn((8, 512), generator=gen, dtype=torch.bfloat16)
    logits_cpu[0, 10] = NEG_INF

    seen_bool = np.zeros((8, 512), dtype=np.bool_)
    seen_bool[0, [3, 10, 31, 42]] = True
    seen_bool[2, [100, 511]] = True
    seen_bitmask_np = _pack_bool_mask(seen_bool)
    penalties_np = np.ones((8, 1), dtype=np.float32)
    penalties_np[0, 0] = 1.05
    penalties_np[2, 0] = 1.5

    with torch.compiler.set_stance("force_eager"):
        ref = rep_pen.apply_penalties_table(logits_cpu, seen_bitmask_np, penalties_np)

    got = rep_pen.apply_penalties_table(
        logits_cpu.to(device), seen_bitmask_np, penalties_np
    ).cpu()

    assert got.dtype == torch.bfloat16
    assert torch.equal(got, ref)
