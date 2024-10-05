import pytest
import socket
from contextlib import contextmanager
import os
from unittest import mock


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
    def dist_environment(
        random_available_port, backend=None, rank=0, local_rank=0, world_size=1, local_world_size=1, global_unique_id=""
    ):
        with mock.patch.dict(
            os.environ,
            {
                "GLOBAL_UNIQUE_ID": global_unique_id,
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(local_rank),
                "LOCAL_WORLD_SIZE": str(local_world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(random_available_port),
                "ZERO_BAND_LOG_LEVEL": "DEBUG",
            },
        ):
            yield

    return dist_environment
