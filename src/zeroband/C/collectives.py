from typing import Optional
import torch
import torch.distributed as dist
from torch.utils import cpp_extension
from pathlib import Path

INCLUDES = [str(Path(__file__).parent / "csrc"), "/home/jackmin/Documents/ZeroBand/third_party/gloo"]
COLLECTIVES_CSRC_PATH = Path(__file__).parent / "csrc" / "collectives.cpp"

collectives_ops = cpp_extension.load(
    name="collectives",
    sources=[COLLECTIVES_CSRC_PATH],
    extra_cflags=["-O3", "-DUSE_C10D_GLOO"],
    verbose=True,
    extra_include_paths=INCLUDES,
)


def ring_allreduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if isinstance(group, dist.distributed_c10d.ProcessGroupGloo):
        collectives_ops.ring_allreduce_gloo(tensor, op, group)
    else:
        collectives_ops.ring_allreduce(tensor, op, group)
