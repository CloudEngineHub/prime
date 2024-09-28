"""
torch distribted test

this test are different from the torchrun integration tests

They manually do the job of torchrun to start the distributed process making it easy to write unit tests
"""

import torch.distributed as dist
import torch
import pytest

import multiprocessing


@pytest.mark.parametrize("world_size", [2])
def test_all_reduce(world_size, random_available_port, dist_environment):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, rank=rank, world_size=world_size):
            data = (rank + 1) * torch.ones(10, 10).to(f"cuda:{rank}")
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            assert data.mean() == sum([i + 1 for i in range(world_size)])

    processes = [multiprocessing.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
