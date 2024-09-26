"""
torch distribted test

this test are different from the torchrun integration tests

They manually do the job of torchrun to start the distributed process making it easy to write unit tests
"""

import torch
import pytest
from torch.distributed import destroy_process_group, init_process_group


import os
from unittest import mock
import socket
from contextlib import contextmanager
import gc


@pytest.fixture(autouse=True)
def memory_cleanup():
    # credits to : https://github.com/pytorch/pytorch/issues/82218#issuecomment-1675254117
    try:
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def random_available_port():
    return get_random_available_port()


@pytest.fixture()
def dist_environment() -> callable:
    @contextmanager
    def dist_environment(random_available_port, local_rank=0, world_size=1, local_world_size=1):
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_WORLD_SIZE": str(local_world_size),
                "RANK": str(local_rank),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(random_available_port),
                "ZERO_BAND_LOG_LEVEL": "DEBUG",
            },
        ):
            try:
                init_process_group()
                torch.cuda.set_device(local_rank)
                yield
            finally:
                destroy_process_group()

    return dist_environment
