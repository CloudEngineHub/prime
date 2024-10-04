from typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

COMPRESS_CSRC_PATH = Path(__file__).parent / "csrc" / "compression.cpp"

compress_ops = load(name="compression", sources=[COMPRESS_CSRC_PATH], extra_cflags=["-O3"], verbose=False)


def uniform_8bit_quantize(tensor: torch.Tensor, inplace: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to 8-bit integers
    Args:
        tensor (torch.Tensor): The tensor to quantize
        inplace (bool): Whether the operation is allowed to modify the input tensor
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the lookup table
    """
    return compress_ops.uniform_8bit_quantize(tensor, inplace)


def average_buckets(tensor: torch.Tensor, quant_weight: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Return the average value in each bin
    Args:
        tensor (torch.Tensor): The tensor to average
        quant_weight (torch.Tensor): The tensor of indices
        n_bins (int): The number of bins
    Returns:
        torch.Tensor: The average value in each bin
    """
    return compress_ops.average_buckets(tensor, quant_weight, n_bins)


def quantize_per_tensor_uint8(tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Quantize a tensor to 8-bit integers

    quantized_value = clamp((round(input / scale) + zero_point), 0, 255)

    Args:
        tensor (torch.Tensor): The tensor to quantize
        scale (float): The scale of the quantization
        zero_point (int): The zero point of the quantization
    Returns:
        torch.Tensor: The quantized tensor
    """
    return compress_ops.quantize_per_tensor_uint8(tensor, scale, zero_point)
