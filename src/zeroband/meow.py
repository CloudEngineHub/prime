import os

import torch
import torch.distributed as dist

from zeroband.comms import ElasticDeviceMesh

from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger


def train():
    elastic_device_mesh = ElasticDeviceMesh()
    # dist.init_process_group(backend="gloo")
    # group = dist.distributed_c10d._get_default_group()
    group = elastic_device_mesh.global_pg

    logger.info(f"rank: {group.rank()}")

    data = torch.ones(10, 10) * world_info.local_rank
    dist.all_reduce(data, op=dist.ReduceOp.SUM, group=group)
    logger.info(msg=f"data: {data.mean() / elastic_device_mesh.global_pg.size()}")

    # logger.info(f"global rank: {world_info.global_rank}")

    if world_info.local_rank == 1:
        dest_rank = 0
        logger.info(f"Sending param {data.shape} to {dest_rank}")
        group.send([data], dest_rank, 0).wait()

    if world_info.local_rank == 0:
        src_rank = 1
        logger.info(f"Receiving param {data.shape} from {src_rank}")
        group.recv([data], src_rank, 0).wait()

    # logger.info(f"data: {data.mean()}")
    logger.info("finish")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)  # this ensure same weight init across diloco workers

    world_info = get_world_info()
    logger = get_logger()

    # torch.cuda.set_device(world_info.local_rank)

    train()
