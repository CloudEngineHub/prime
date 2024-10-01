import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load
# Define the C++ code as a string


dist_ops = load(name="extension", sources=["src/zeroband/hello.cpp"], extra_cflags=["-O2"], verbose=True)


def main():
    dist.init_process_group(backend="gloo")
    group = dist.distributed_c10d._get_default_group()
    rank = dist.get_rank()
    m = torch.ones(10) * rank

    print(f"rank {rank}: {m}")
    dist_ops.send_recv(m, group)
    print(f"rank {rank}: {m}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
