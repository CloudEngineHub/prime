from pydantic_config import BaseConfig, parse_argv
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.utils.benchmark as benchmark

from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger


class Config(BaseConfig):
    size_model: int = int(1e9)
    n_iters: int = 5


def main(config: Config):
    world_info = get_world_info()

    mat = torch.rand(1, config.size_model)

    logger.info(
        f"\n ======== Benchmark all reduce between {world_info.world_size} gpus over {world_info.nnodes} nodes =========\n"
    )

    t0 = benchmark.Timer(stmt="dist.all_reduce(mat)", setup="from __main__ import dist", globals={"mat": mat})
    measured_time = t0.timeit(config.n_iters).mean

    bandwidth = config.size_model * 4 / 1e9 / measured_time

    logger.info(f"Average time per iteration: {measured_time:.2f} seconds, Average bandwidth: {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    config = Config(**parse_argv())

    torch.set_float32_matmul_precision("high")
    init_process_group(backend="gloo")

    logger = get_logger()
    main(config)
    destroy_process_group()
