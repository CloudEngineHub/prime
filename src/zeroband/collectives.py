from enum import Enum
from typing import Callable, Optional, TypeAlias
import torch
import torch.distributed as dist

AllReduceFunc: TypeAlias = Callable[
    [torch.Tensor, dist.ReduceOp, Optional[dist.ProcessGroup], Optional[torch.dtype]], None
]


def gloo_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Wrap gloo all reduce"""
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if op not in [dist.ReduceOp.SUM, dist.ReduceOp.AVG]:
        raise ValueError(f"Unsupported reduce operation {op}. Only SUM and AVG are supported.")

    # group = cast(dist.ProcessGroup, group) # just type hint stuff for IDE
    if op == dist.ReduceOp.AVG:
        # todo check numerical stability of doing post or pre div
        tensor.div_(group.size())

    dist.all_reduce(tensor, op, group=group)


class Compression(Enum):
    NO = "no"
    UINT8 = "uint8"


def all_reduce(
    compression: Compression,
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    if compression == Compression.UINT8:
        from zeroband.C.collectives import ring_allreduce as ring_allreduce_c

        return ring_allreduce_c(tensor, op, group)
    else:
        return gloo_all_reduce(tensor, op, group)
