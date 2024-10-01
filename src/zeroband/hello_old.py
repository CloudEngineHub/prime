import torch
import torch.distributed as dist


def send_recv(
    tensor: torch.Tensor,
) -> None:
    """
    Send and receive a tensor between two processes.
    """
    rank = torch.distributed.get_rank()

    if rank == 1:
        torch.distributed.send(tensor, dst=0)
    else:
        torch.distributed.recv(tensor, src=1)


dist.init_process_group(backend="gloo")


rank = torch.distributed.get_rank()
m = torch.ones(10) * rank


print(f"rank {rank}: {m}")
send_recv(m)
print(f"rank {rank}: {m}")

dist.destroy_process_group()
