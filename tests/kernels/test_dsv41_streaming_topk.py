# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-row prefill streaming top-k: stable ties and finite padding contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    streaming_topk,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.tuned_block_sizes import (
    STREAMING_TOPK_CONFIG,
    StreamingTopKConfig,
    streaming_topk_config,
)

NEAREST_CONFIG = "TPU_DEEPSEEK_V41_STREAMING_TOPK_NEAREST_CONFIG"


@pytest.mark.parametrize("block,threshold", [(8192, 8192), (16384, 32768)])
@pytest.mark.parametrize("width", [65536, 65408])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "masked",
    ],
)
def test_streaming_topk_contract(block, threshold, width, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(32, width)).astype(np.float32)
    if distribution == "ascending":
        scores[:] = np.arange(width, dtype=np.float32)
    elif distribution == "descending":
        scores[:] = -np.arange(width, dtype=np.float32)
    elif distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        # Equal Kth values survive repeated candidate compactions, with later
        # blocks also supplying larger winners that displace earlier ties.
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:4] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 16383, 50176, width], np.int32), 32
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            block_size=block,
            reconstruct_threshold=threshold,
        )
    )
    for row, length in enumerate(lengths):
        ids = actual[row]
        ids = ids[(ids >= 0) & (scores[row, np.maximum(ids, 0)] > -np.inf)]
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        order = np.lexsort((eligible, ~keys))[:512]
        np.testing.assert_array_equal(np.sort(ids), np.sort(eligible[order]))
        assert np.unique(ids).size == ids.size


@pytest.mark.parametrize(
    ("width", "nearest", "measured"),
    [(98816, False, 98816), (59904, False, None), (59904, True, 49408)]
    + [(419328, True, 98816)],
    ids=["measured", "unmeasured_default", "nearest_below", "nearest_above"],
)
def test_streaming_topk_config_of_unmeasured_widths(
    monkeypatch, width, nearest, measured
):
    """Only the flag lets a width without an entry borrow a measured config."""
    monkeypatch.setenv(NEAREST_CONFIG, "1" if nearest else "0")
    expected = (
        StreamingTopKConfig()
        if measured is None
        else STREAMING_TOPK_CONFIG[(7, measured, 512)]
    )
    assert streaming_topk_config(7, width, 512) == expected


@pytest.mark.parametrize(
    "nearest", [False, True], ids=["default_config", "nearest_config"]
)
def test_unordered_topk_scores_at_an_unmeasured_width(monkeypatch, nearest):
    """59,904 keys, one key tile of an 835,584-token context: same top scores."""
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    monkeypatch.setenv(NEAREST_CONFIG, "1" if nearest else "0")
    width, rows = 59904, 32
    rng = np.random.default_rng(23)
    # Indexer scores are sums of ReLU outputs: non-negative, with exact zeros.
    scores = np.maximum(rng.normal(size=(rows, width)), 0).astype(np.float32)
    lengths = np.resize(np.array([513, 4096, 20000, 40000, width], np.int32), rows)
    # A fresh trace reads the flag; the public entry point caches its own.
    select = jax.jit(
        lambda scores, lengths: streaming_topk.__wrapped__(
            scores,
            valid_lengths=lengths,
            k=512,
            exclude_neg_inf=True,
            reconstruct_threshold=2048,
            unordered=True,
        )
    )
    actual = np.asarray(select(jnp.asarray(scores), jnp.asarray(lengths)))
    for row, length in enumerate(lengths):
        ids = actual[row][actual[row] >= 0]
        assert np.unique(ids).size == ids.size == 512
        assert (ids < length).all()
        np.testing.assert_array_equal(
            np.sort(scores[row, ids]), np.sort(scores[row, :length])[-512:]
        )
