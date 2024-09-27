"""test Diloco."""

import multiprocessing
import pytest

import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from zeroband.diloco import Diloco, DilocoConfig


@pytest.mark.parametrize("world_size", [2])  # [1, 2])
def test_diloco_all_reduce(world_size, random_available_port, dist_environment):
    """
    In this test we manually create a inner model and a outer model where we control the weight:
    inner has weight: (rank + 1) / 2
    outer has weight: (rank + 1)

    since we know the world_size we can predict the results of the all reduce of the pseudo gradient and therefore test
    if it is done correclty.
    """

    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, local_rank=rank, world_size=world_size):
            diloco_config = DilocoConfig(inner_steps=10)

            model = torch.nn.Linear(10, 10)

            # init param to rank + 1
            for param in model.parameters():
                param.data = (rank + 1) * torch.ones_like(param.data).to("cuda")

            global_pg = dist.new_group(backend="gloo")
            diloco = Diloco(diloco_config, model, ShardingStrategy.FULL_SHARD, global_pg)

            # simulate inner model updates
            for param in model.parameters():
                param.data = (rank + 1) / 2 * torch.ones_like(param.data).to("cuda")

            diloco.sync_pseudo_gradient(model)

            for param in diloco.param_list_cpu:
                print(f"param.grad.mean() {param.grad.mean()}")
                target = (
                    torch.ones_like(param.grad)
                    * sum([(rank + 1) - (rank + 1) / 2 for rank in range(world_size)])
                    / world_size
                )
                assert param.grad.mean() == target.mean()

    processes = [multiprocessing.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
