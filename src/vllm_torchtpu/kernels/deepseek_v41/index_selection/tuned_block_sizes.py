# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm_torchtpu import envs

# (TPU generation, score width, candidate count, records per candidate block).
PUBLISH_CANDIDATE_SHAPES = frozenset({(7, 16384, 2048, 8)})
# (generation, queries, token-score width, block size): (rows, output blocks).
PUBLISH_BLOCK_MAX_CONFIG = {(7, 96, 98816, 8): (96, 1024)}


@dataclass(frozen=True)
class StreamingTopKConfig:
    loop_unroll: int = 1
    bootstrap_multiplier: int = 4
    bitonic_ids: bool = False


# (generation, score width, topk): measured scan, bootstrap and output sorting.
STREAMING_TOPK_CONFIG = {
    (7, width, 512): StreamingTopKConfig(4, 8, True)
    for width in (16384, 32768, 49408, 98816)
}


def streaming_topk_config(generation: int, width: int, k: int) -> StreamingTopKConfig:
    """The measured config of this shape, or the untuned default.

    With TPU_DEEPSEEK_V41_STREAMING_TOPK_NEAREST_CONFIG, a score width without
    a measured entry takes the entry of the nearest measured width of the same
    generation and k (835,584 tokens of context score 59,904 or 419,328 keys,
    neither of which is measured) instead of the default.
    """
    config = STREAMING_TOPK_CONFIG.get((generation, width, k))
    if config is not None:
        return config
    if envs.TPU_DEEPSEEK_V41_STREAMING_TOPK_NEAREST_CONFIG:
        widths = [
            measured
            for tuned_generation, measured, tuned_k in STREAMING_TOPK_CONFIG
            if (tuned_generation, tuned_k) == (generation, k)
        ]
        if widths:
            nearest = min(widths, key=lambda measured: abs(measured - width))
            return STREAMING_TOPK_CONFIG[(generation, nearest, k)]
    return StreamingTopKConfig()
