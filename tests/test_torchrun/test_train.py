import pickle
import subprocess
import numpy as np
import pytest
import socket


def get_random_available_port():
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def random_available_port():
    return get_random_available_port()



@pytest.fixture()
def config_path() -> str:
    # need to be executed in the root dir
    return "configs/debug.toml" 



@pytest.mark.parametrize("num_gpu", [1, 2])
def test_multi_gpu_ckpt(config_path, random_available_port, num_gpu):

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpu}",
        "--rdzv-endpoint",
        f"localhost:{random_available_port}",
        "src/zeroband/train.py",
        f"@{config_path}",
        "--optim.total_steps",
        "10"
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        pytest.fail(f"Process {result} failed {result.stderr}")