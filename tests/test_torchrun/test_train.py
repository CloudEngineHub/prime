import copy
import os
from pathlib import Path
import shutil
import subprocess
import pytest
import socket


def get_random_available_port_list(num_port):
    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    ports = []

    while len(ports) < num_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            new_port = s.getsockname()[1]

        if new_port not in ports:
            ports.append(new_port)

    return ports


def get_random_available_port(num_port):
    return get_random_available_port_list(num_port)[0]


def gpus_to_use(num_nodes, num_gpu, rank):
    return ",".join(map(str, range(rank * num_gpu, (rank + 1) * num_gpu)))


def _test_multi_gpu(num_gpus, config, extra_args=[]):
    num_nodes, num_gpu = num_gpus[0], num_gpus[1]

    processes = []
    ports = get_random_available_port_list(num_nodes)
    for i in range(num_nodes):
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpu}",
            "--rdzv-endpoint",
            f"localhost:{ports[i]}",
            "src/zeroband/train.py",
            f"@configs/{config}",
            *extra_args,
        ]

        env = copy.deepcopy(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus_to_use(num_nodes, num_gpu, i)
        process1 = subprocess.Popen(cmd, env=env)
        processes.append(process1)

    for process in processes:
        result = process.wait()
        if result != 0:
            pytest.fail(f"Process {result} failed {result}")


@pytest.mark.parametrize("num_gpus", [[1, 1], [2, 1], [1, 2]])
def test_multi_gpu(num_gpus):
    _test_multi_gpu(num_gpus, "debug/normal.toml")


@pytest.mark.parametrize("num_gpus", [[1, 2], [2, 2]])
def test_multi_gpu_diloco(num_gpus):
    # we don't test 1,1 and 2,1 because 1 solo gpu failed with fsdp
    _test_multi_gpu(num_gpus, "debug/diloco.toml")


@pytest.mark.parametrize("strategy", ["SHARD_GRAD_OP"])
def test_multi_gpu_diloco_non_full_shard(strategy):
    # we don't test 1,1 and 2,1 because 1 solo gpu failed with fsdp
    num_gpus = [2, 2]
    _test_multi_gpu(num_gpus, "debug/diloco.toml", extra_args=["--train.sharding_strategy", strategy])


## test ckpt


def test_ckpt(tmp_path: Path):
    ckpt_path = "outputs"  # for some reason tmp_path is not working
    os.makedirs(ckpt_path, exist_ok=True)
    _test_multi_gpu([1, 1], "debug/normal.toml", extra_args=["--ckpt.path", str(ckpt_path), "--ckpt.interval", "10"])
    _test_multi_gpu([1, 1], "debug/normal.toml", extra_args=["--resume", str(ckpt_path)])
    shutil.rmtree(ckpt_path)
