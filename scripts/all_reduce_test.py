from pydantic_config import BaseConfig, parse_argv
import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.utils.benchmark as benchmark

from zeroband.collectives import AllReduceBackend, ALL_REDUCE_FN
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from typing import Optional
from enum import Enum


class TorchDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"


TORCH_DTYPE_MAP = {
    None: None,
    TorchDtype.FLOAT32: torch.float32,
    TorchDtype.FLOAT16: torch.float16,
    TorchDtype.BFLOAT16: torch.bfloat16,
    TorchDtype.UINT8: torch.uint8,
}


class Config(BaseConfig):
    size_model: int = int(1e9)
    n_iters: int = 5
    backend: AllReduceBackend = AllReduceBackend.GLOO
    transfer_dtype: Optional[TorchDtype] = None


def main(config: Config):
    world_info = get_world_info()

    mat = torch.rand(1, config.size_model)

    logger.info(
        f"\n ======== Benchmark all reduce between {world_info.world_size} gpus over {world_info.nnodes} nodes =========\n"
    )

    all_reduce = ALL_REDUCE_FN[config.backend]
    transfer_dtype = TORCH_DTYPE_MAP[config.transfer_dtype]

    if config.transfer_dtype is not None and transfer_dtype.is_floating_point:
        t0 = benchmark.Timer(
            stmt="all_reduce(mat, transfer_dtype=transfer_dtype)",
            globals={"all_reduce": all_reduce, "mat": mat, "transfer_dtype": transfer_dtype},
        )
    elif config.transfer_dtype is not None and torch.uint8:
        from zeroband.C.compression import uniform_8bit_quantize

        t0 = benchmark.Timer(
            stmt="all_reduce(mat, quantization_func=foo)",
            globals={"all_reduce": all_reduce, "mat": mat, "foo": uniform_8bit_quantize},
        )
    else:
        t0 = benchmark.Timer(stmt="all_reduce(mat)", globals={"all_reduce": all_reduce, "mat": mat})
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
