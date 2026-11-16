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
"""``repetition_penalty`` for the TPU sampler.

The TPU runner samples on device without running vLLM's GPU ``Sampler``
penalties pass. vLLM's ``InputBatch`` already keeps the penalty of each row
(``repetition_penalties_cpu``), the requests that have one
(``repetition_penalties_reqs``) and their output ids
(``req_output_token_ids``), and moves them with the rows when the batch is
condensed or reordered. This module reads them from there and keeps only a
seen-token bitmask per request id (``RepetitionPenaltyTracker``).

For each penalized request, ``SeenTokenBitmask`` packs presence bits over
vocabulary ids ``[0, vocab_size)`` from ``prompt_token_ids`` and the output ids
into ``cdiv(vocab_size, 32)`` ``int32`` words (out-of-range ids such as
multimodal placeholders or text tokens wider than a codec head are ignored). On
device, the bitmask is unpacked and seen positions are scaled in ``float32`` by
``1 / penalty`` where ``logits > 0`` and ``penalty`` otherwise before casting
back to ``logits.dtype``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

_BIT_MASKS: np.ndarray = (np.uint32(1) << np.arange(32, dtype=np.uint32)).view(np.int32)


def num_bitmask_words(vocab_size: int) -> int:
    """Number of 32-bit words needed to pack ``vocab_size`` token presence bits."""
    return (vocab_size + 31) >> 5


class SeenTokenBitmask:
    """Packed presence bits of one request's prompt and output ids, updated
    incrementally as the output grows."""

    def __init__(self) -> None:
        self._vocab_size = 0
        self._bitmask: np.ndarray | None = None
        self._prompt_ids: Sequence[int] | None = None
        self._output_ids: Sequence[int] | None = None
        self._committed_len = 0

    def update(
        self,
        prompt_ids: Sequence[int] | None,
        output_ids: Sequence[int],
        vocab_size: int,
    ) -> np.ndarray:
        """Return the ``int32`` ``[cdiv(vocab_size, 32)]`` presence bitmask over
        ``prompt_ids`` and ``output_ids``.

        The bitmask is built again when the request's prompt or output list is
        a different object (a streaming-input continuation, or a finished id
        sent again) or the output became shorter.
        """
        num_words = num_bitmask_words(vocab_size)
        cur_len = len(output_ids)
        if (
            self._bitmask is None
            or self._vocab_size != vocab_size
            or prompt_ids is not self._prompt_ids
            or output_ids is not self._output_ids
            or cur_len < self._committed_len
        ):
            if self._bitmask is None or self._bitmask.shape[0] != num_words:
                self._bitmask = np.zeros((num_words,), dtype=np.int32)
            else:
                self._bitmask.fill(0)
            if prompt_ids is not None and len(prompt_ids) > 0:
                prompt = np.asarray(prompt_ids, dtype=np.int64)
                valid = prompt[(prompt >= 0) & (prompt < vocab_size)]
                np.bitwise_or.at(self._bitmask, valid >> 5, _BIT_MASKS[valid & 31])
            self._vocab_size = vocab_size
            self._prompt_ids = prompt_ids
            self._output_ids = output_ids
            self._committed_len = 0

        if cur_len > self._committed_len:
            bitmask = self._bitmask
            for tok in output_ids[self._committed_len : cur_len]:
                if 0 <= tok < vocab_size:
                    bitmask[tok >> 5] |= _BIT_MASKS[tok & 31]
            self._committed_len = cur_len

        return self._bitmask


class RepetitionPenaltyTracker:
    """Seen-token bitmasks, by request id, of the requests that
    ``InputBatch.repetition_penalties_reqs`` lists."""

    def __init__(self) -> None:
        self._seen: dict[str, SeenTokenBitmask] = {}
        self._bitmask_buf: np.ndarray | None = None
        self._penalties_buf: np.ndarray | None = None
        self._dirty_rows: list[int] = []

    def prune(self, req_ids: Collection[str]) -> None:
        """Drop the bitmasks of requests that are not in ``req_ids``."""
        if len(self._seen) > len(req_ids):
            for req_id in [r for r in self._seen if r not in req_ids]:
                del self._seen[req_id]

    def build_plain_inputs(
        self,
        input_batch: InputBatch,
        requests: Mapping[str, Any],
        start: int,
        end: int,
        num_rows: int,
        vocab_size: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stage ``(seen_bitmask, penalties)`` for the plain 1-row/req logits
        chunk of batch rows ``[start, end)`` into reusable host buffers, or
        return ``None`` when no row in the chunk has a penalty."""
        req_id_to_index = input_batch.req_id_to_index
        indices = [
            index
            for req_id in input_batch.repetition_penalties_reqs
            if start <= (index := req_id_to_index[req_id]) < end
        ]
        if not indices or vocab_size <= 0:
            return None
        num_words = num_bitmask_words(vocab_size)
        if (
            self._bitmask_buf is None
            or self._penalties_buf is None
            or self._bitmask_buf.shape[0] < num_rows
            or self._bitmask_buf.shape[1] != num_words
        ):
            self._bitmask_buf = np.zeros((num_rows, num_words), dtype=np.int32)
            self._penalties_buf = np.ones((num_rows, 1), dtype=np.float32)
            self._dirty_rows.clear()
        else:
            for prev_row in self._dirty_rows:
                self._bitmask_buf[prev_row].fill(0)
                self._penalties_buf[prev_row, 0] = 1.0
            self._dirty_rows.clear()

        for batch_idx in indices:
            row = batch_idx - start
            req_id = input_batch.req_ids[batch_idx]
            seen = self._seen.get(req_id)
            if seen is None:
                seen = self._seen[req_id] = SeenTokenBitmask()
            output_ids = input_batch.req_output_token_ids[batch_idx]
            assert req_id is not None and output_ids is not None
            self._bitmask_buf[row] = seen.update(
                requests[req_id].prompt_token_ids, output_ids, vocab_size
            )
            self._penalties_buf[row, 0] = input_batch.repetition_penalties_cpu[
                batch_idx
            ]
            self._dirty_rows.append(row)
        if self._bitmask_buf.shape[0] == num_rows:
            return self._bitmask_buf, self._penalties_buf
        return self._bitmask_buf[:num_rows], self._penalties_buf[:num_rows]


def apply_penalties_table(
    logits: torch.Tensor,
    seen_bitmask_np: np.ndarray,
    penalties_np: np.ndarray,
) -> torch.Tensor:
    """Stage host packed bitmask and penalty tables to ``logits.device`` and apply
    scaling."""
    device = logits.device
    seen_bitmask = torch.from_numpy(seen_bitmask_np).to(device)
    penalties = torch.from_numpy(penalties_np).to(device)
    return apply_repetition_penalty(logits, seen_bitmask, penalties)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def apply_repetition_penalty(
    logits: torch.Tensor,
    seen_bitmask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    return _apply_repetition_penalty(logits, seen_bitmask, penalties)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_bitmask: torch.Tensor,
    penalties: torch.Tensor,
) -> torch.Tensor:
    """Unpack ``seen_bitmask`` (``[rows, cdiv(vocab_size, 32)]`` ``int32``) and
    scale ``logits`` in ``float32`` by ``1 / penalties`` where ``logits > 0`` and
    ``penalties`` otherwise on seen positions, preserving ``logits.dtype``."""
    target_dim = logits.shape[-1]
    shifts = torch.arange(32, dtype=torch.int32, device=seen_bitmask.device)
    seen_mask = (
        torch.bitwise_right_shift(seen_bitmask[:, :, None], shifts[None, None, :]) & 1
    ) != 0
    seen_mask = seen_mask.reshape(logits.shape[0], -1)[:, :target_dim]
    logits_f32 = logits.to(torch.float32)
    inv_penalties = 1.0 / penalties
    pos_scale = torch.where(logits_f32 > 0, inv_penalties, penalties)
    scaling = torch.where(seen_mask, pos_scale, 1.0)
    return (logits_f32 * scaling).to(logits.dtype)
