from typing import Optional, Callable
import torch
import torch.distributed as dist
from torch.utils import cpp_extension
from pathlib import Path

INCLUDES = [str(Path(__file__).parent / "csrc")]
COLLECTIVES_CSRC_PATH = Path(__file__).parent / "csrc" / "collectives.cpp"

collectives_ops = cpp_extension.load(
    name="collectives",
    sources=[COLLECTIVES_CSRC_PATH],
    extra_cflags=["-O3"],
    verbose=True,
    extra_include_paths=INCLUDES,
)


def ring_allreduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    transfer_dtype: Optional[torch.dtype] = None,
    quantization_func: Optional[Callable] = None,
) -> None:
    collectives_ops.ring_allreduce(tensor, op, group)
