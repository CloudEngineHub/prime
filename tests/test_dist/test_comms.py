import time
import torch
import torch.distributed as dist
import pytest
from zeroband.comms import ElasticDeviceMesh
import multiprocessing as mp


@pytest.mark.parametrize("world_size", [2, 8])
def test_elastic_device_mesh_no_global(world_size: int, random_available_port: int, mock_env):
    def foo(**kwargs):
        with mock_env(**kwargs):
            edm = ElasticDeviceMesh()

            rank = int(kwargs["RANK"])
            a = torch.arange(3) * (rank + 1)
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.local_pg)
            sum_ints = world_size * (world_size + 1) // 2
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.global_pg)
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

    processes = []
    for rank in range(world_size):
        processes.append(
            mp.Process(
                target=foo,
                kwargs={
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(random_available_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": str(world_size),
                    "ZERO_BAND_LOG_LEVEL": "DEBUG",
                },
            )
        )
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("global_world_size", [2, 8])
def test_elastic_device_mesh(world_size: int, global_world_size: int, mock_env):
    def foo(**kwargs):
        with mock_env(**kwargs):
            edm = ElasticDeviceMesh()

            rank = int(kwargs["RANK"])
            a = torch.arange(3) * (rank + 1)
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.local_pg)
            sum_ints = world_size * (world_size + 1) // 2
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

            global_rank = int(kwargs["GLOBAL_RANK"])
            a = torch.arange(3) * (global_rank + 1)
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.global_pg)
            sum_ints = global_world_size * (global_world_size + 1) // 2
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

    global_ports = [i for i in range(21970, 21970 + world_size)]
    master_ports = [i for i in range(31000, 31000 + global_world_size)]
    processes = []
    for global_rank in range(global_world_size):
        for rank in range(world_size):
            processes.append(
                mp.Process(
                    target=foo,
                    kwargs={
                        "MASTER_ADDR": "localhost",
                        "MASTER_PORT": str(master_ports[global_rank]),
                        "RANK": str(rank),
                        "WORLD_SIZE": str(world_size),
                        "LOCAL_RANK": str(rank),
                        "LOCAL_WORLD_SIZE": str(world_size),
                        "GLOBAL_UNIQUE_ID": str(global_rank),
                        "GLOBAL_ADDR": "localhost",
                        "GLOBAL_PORT": str(global_ports[0]),
                        "GLOBAL_RANK": str(global_rank),
                        "GLOBAL_WORLD_SIZE": str(global_world_size),
                        "ZERO_BAND_LOG_LEVEL": "DEBUG",
                    },
                )
            )
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("global_world_size", [2, 8])
def test_elastic_device_mesh_on_ramp(world_size: int, global_world_size: int, mock_env):
    def foo(**kwargs):
        with mock_env(**kwargs):
            test_value = int(kwargs["TEST_VALUE"])

            edm = ElasticDeviceMesh()
            edm.maybe_reinit_global_pg()
            assert edm.mesh_count == 0
            assert edm.global_pg.size() == global_world_size
            time.sleep(1)
            edm.maybe_reinit_global_pg()
            assert edm.mesh_count == 1
            assert edm.global_pg.size() == global_world_size + 1

            a = torch.arange(3) * (test_value + 1)
            sum_ints = global_world_size * (global_world_size + 1) // 2 + 100
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.global_pg)
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

            dist.barrier(edm.global_pg)

    def bar(**kwargs):
        with mock_env(**kwargs):
            test_value = int(kwargs["TEST_VALUE"])
            time.sleep(1)  # TODO: Use queue to signal instead of time

            edm = ElasticDeviceMesh()
            assert edm.mesh_count == 1
            assert edm.global_pg.size() == global_world_size + 1

            a = torch.arange(3) * test_value
            sum_ints = global_world_size * (global_world_size + 1) // 2 + 100
            dist.all_reduce(a, op=dist.ReduceOp.SUM, group=edm.global_pg)
            assert torch.allclose(a, torch.tensor([0, sum_ints, 2 * sum_ints]))

            dist.barrier(edm.global_pg)

    global_ports = [i for i in range(21970, 21970 + world_size)]
    master_ports = [i for i in range(31000, 31000 + global_world_size + 1)]
    processes = []
    for global_rank in range(global_world_size):
        for rank in range(world_size):
            processes.append(
                mp.Process(
                    target=foo,
                    kwargs={
                        "MASTER_ADDR": "localhost",
                        "MASTER_PORT": str(master_ports[global_rank]),
                        "RANK": str(rank),
                        "WORLD_SIZE": str(world_size),
                        "LOCAL_RANK": str(rank),
                        "LOCAL_WORLD_SIZE": str(world_size),
                        "GLOBAL_UNIQUE_ID": str(global_rank),
                        "GLOBAL_ADDR": "localhost",
                        "GLOBAL_PORT": str(global_ports[0]),
                        "GLOBAL_RANK": str(global_rank),
                        "GLOBAL_WORLD_SIZE": str(global_world_size),
                        "ZERO_BAND_LOG_LEVEL": "DEBUG",
                        "TEST_VALUE": str(global_rank),
                    },
                )
            )

    for rank in range(world_size):
        processes.append(
            mp.Process(
                target=bar,
                kwargs={
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(master_ports[global_world_size]),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": str(world_size),
                    "GLOBAL_UNIQUE_ID": "A",
                    "GLOBAL_ADDR": "localhost",
                    "GLOBAL_PORT": str(global_ports[0]),
                    "GLOBAL_RANK": "100",
                    "GLOBAL_WORLD_SIZE": str(global_world_size),
                    "ZERO_BAND_LOG_LEVEL": "DEBUG",
                    "TEST_VALUE": "100",
                },
            )
        )

    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
