# Code adapted from https://github.com/PrimeIntellect-ai/hivemind/blob/213bff98a62accb91f254e2afdccbf1d69ebdea9/hivemind/compression/quantization.py
# Original code is licensed under the MIT License.
# See the LICENSE file in the original repository for more information.

import torch
import numpy as np
from typing import Tuple
import math
from concurrent.futures import ThreadPoolExecutor
import os

RANGE_IN_SIGMAS: int = 6
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("QUANTIZATION_THREADS", 128)))
n_bins = 2**8


def average_buckets(tensor: torch.Tensor, quant_weight: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Return the average value in each bucket"""
    bin_sums = torch.zeros(n_bins).scatter_add_(0, quant_weight.flatten().long(), tensor.flatten())
    bin_counts = torch.clamp_min_(torch.bincount(quant_weight.flatten(), minlength=n_bins), 1)
    lookup = bin_sums / bin_counts
    return lookup


def get_chunk_size(num_elements: int, min_chunk_size: int) -> int:
    """Adjust chunk_size to minimize imbalance between chunk sizes"""
    if min_chunk_size >= num_elements:
        return min_chunk_size
    leftover_elements = num_elements % min_chunk_size
    num_chunks = num_elements // min_chunk_size
    return min_chunk_size + (leftover_elements - 1) // num_chunks + 1


def quantile_qq_approximation(array: np.ndarray, n_quantiles: int, min_chunk_size: int = 10**5) -> np.ndarray:
    """Estimate uniform quantiles of data using quantile-of-quantiles. Runs in parallel."""
    if not array.data.c_contiguous and array.data.f_contiguous:
        array = array.T
    array = np.ascontiguousarray(array.reshape(-1))
    quantiles = np.linspace(0.0, 1.0, num=n_quantiles, dtype=array.dtype)
    chunk_size = get_chunk_size(len(array), min_chunk_size)
    num_chunks = (len(array) - 1) // chunk_size + 1
    partition_quantiles = np.empty((num_chunks, len(quantiles)), dtype=array.dtype)

    jobs = []
    for i in range(num_chunks):
        chunk = slice(chunk_size * i, chunk_size * (i + 1))
        jobs.append(EXECUTOR.submit(np.quantile, array[chunk], quantiles, out=partition_quantiles[i]))

    for job in jobs:
        job.result()
    return np.quantile(partition_quantiles, quantiles)


def uniform_8bit_quantize(tensor: torch.Tensor, inplace: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    offset = n_bins // 2
    # shift = tensor.mean()
    # centered_tensor = tensor.sub_(shift) if inplace else tensor - shift
    centered_tensor = tensor
    std_unbiased = centered_tensor.norm() / math.sqrt(centered_tensor.numel() - 1)
    scale = RANGE_IN_SIGMAS * std_unbiased / n_bins
    quantized = torch.quantize_per_tensor(centered_tensor, scale, offset, torch.quint8).int_repr()
    lookup = average_buckets(tensor, quantized, n_bins)
    return quantized, lookup


def quantile_8bit_quantize(tensor: torch.Tensor, inplace: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    borders = torch.as_tensor(quantile_qq_approximation(tensor.numpy(), n_bins + 1)[1:-1])
    quantized = torch.clamp_(torch.bucketize(tensor, borders), 0, n_bins - 1)
    lookup = average_buckets(tensor, quantized, n_bins)
    return quantized, lookup
