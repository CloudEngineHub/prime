from typing import Optional
import torch
import torch.distributed as dist


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
