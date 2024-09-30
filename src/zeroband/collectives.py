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
    transfer_dtype: Optional[torch.dtype] = None,
) -> None:
    """Wrap gloo all reduce"""
    # # if transfer_dtype is None:
    # #     transfer_dtype = tensor.dtype
    # # if group is None:
    # #     group = dist.distributed_c10d._get_default_group()
    # if op not in [dist.ReduceOp.SUM, dist.ReduceOp.AVG]:
    #     raise ValueError(f"Unsupported reduce operation {op}. Only SUM and AVG are supported.")

    # # group = cast(dist.ProcessGroup, group) # just type hint stuff for IDE
    # if op == dist.ReduceOp.AVG:
    #     # todo check numerical stability of doing post or pre div
    #     tensor.div_(group.size())

    # if group is None:
    #     dist.all_reduce(tensor, op)
    # else:
    #     group.allreduce(tensor, op)

    # todo: investigate why test are failing if we use the pass group
    dist.all_reduce(tensor, op)


def ring_allreduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    transfer_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Perform all-reduce on a tensor using ring algorithm.
    The accumulation will be done in-place on the input tensor.
    The transfers will be done using the specified transfer_dtype.
    """
    if transfer_dtype is None:
        transfer_dtype = tensor.dtype
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if op not in [dist.ReduceOp.SUM, dist.ReduceOp.AVG]:
        raise ValueError(f"Unsupported reduce operation {op}. Only SUM and AVG are supported.")

    world_size = group.size()
    rank = group.rank()

    # Divide the tensor into chunks
    chunks = tensor.chunk(world_size)

    # Temporary buffers for transferring data
    send_buffer = torch.empty_like(chunks[0], dtype=transfer_dtype)
    recv_buffer = torch.empty_like(chunks[0], dtype=transfer_dtype)

    for step in range(world_size - 1):
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size

        send_chunk = (rank - step - 1) % world_size
        recv_chunk = (rank - step - 2) % world_size

        send_buffer.copy_(chunks[send_chunk])
        # Send and receive
        work0 = dist.isend(send_buffer, dst=send_rank, group=group)
        work1 = dist.irecv(recv_buffer, src=recv_rank, group=group)

        work0.wait()
        work1.wait()

        # Update the corresponding chunk
        chunks[recv_chunk].add_(recv_buffer)

    if op == dist.ReduceOp.AVG:
        chunks[rank].divide_(world_size)

    for step in range(world_size - 1):
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size

        send_chunk = (rank - step) % world_size
        recv_chunk = (rank - step - 1) % world_size

        send_buffer.copy_(chunks[send_chunk])
        # Send and receive
        work0 = dist.isend(send_buffer, dst=send_rank, group=group)
        work1 = dist.irecv(recv_buffer, src=recv_rank, group=group)

        work0.wait()
        work1.wait()

        # Update the corresponding chunk
        chunks[recv_chunk].copy_(recv_buffer)


class AllReduceBackend(Enum):
    GLOO = "gloo"
    CUSTOM = "custom"


all_reduce_fn: dict[AllReduceBackend, AllReduceFunc] = {
    AllReduceBackend.GLOO: gloo_all_reduce,
    AllReduceBackend.CUSTOM: ring_allreduce,
}
