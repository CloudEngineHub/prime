import torch
import torch.distributed as dist
import multiprocessing as mp
import pytest

from zeroband.collectives import ring_allreduce
from zeroband.collectives import AllReduceBackend, ALL_REDUCE_FN
from zeroband.compression import uniform_8bit_quantize


@pytest.mark.parametrize("world_size", [2, 3, 8])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.AVG])
def test_ring_allreduce(world_size: int, op: dist.ReduceOp, random_available_port: int, dist_environment):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, "gloo", rank=rank, world_size=world_size):
            world_size = dist.get_world_size()

            # Create a sample tensor
            tensor = torch.randn(world_size - 1, world_size * 8, dtype=torch.float32)
            expected = tensor.clone()

            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            if op == dist.ReduceOp.AVG:
                expected /= world_size
            ring_allreduce(tensor, op=op, group=dist.distributed_c10d._get_default_group())

            assert torch.allclose(tensor, expected)

    # Perform ring all-reduce
    processes = [mp.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.AVG])
@pytest.mark.parametrize("transfer_dtype", [torch.bfloat16])
def test_ring_allreduce_low_precision(
    world_size: int, op: dist.ReduceOp, transfer_dtype: torch.dtype, random_available_port: int, dist_environment
):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, "gloo", rank=rank, world_size=world_size):
            world_size = dist.get_world_size()

            # Create a sample tensor
            tensor = torch.randn(world_size - 1, world_size * 8, dtype=torch.float32)
            expected = tensor.clone()

            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            if op == dist.ReduceOp.AVG:
                expected /= world_size
            ring_allreduce(
                tensor, op=op, group=dist.distributed_c10d._get_default_group(), transfer_dtype=transfer_dtype
            )

            assert torch.allclose(tensor, expected, atol=1e-2, rtol=1e-2)

    # Perform ring all-reduce
    processes = [mp.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.AVG])
def test_ring_allreduce_quantized(world_size: int, op: dist.ReduceOp, random_available_port: int, dist_environment):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, "gloo", rank=rank, world_size=world_size):
            world_size = dist.get_world_size()

            # Create a sample tensor
            tensor = torch.randn(world_size - 1, world_size * 8, dtype=torch.float32)
            expected = tensor.clone()

            dist.all_reduce(expected, op=dist.ReduceOp.SUM)
            if op == dist.ReduceOp.AVG:
                expected /= world_size
            ring_allreduce(
                tensor, op=op, group=dist.distributed_c10d._get_default_group(), quantization_func=uniform_8bit_quantize
            )

            error = (tensor - expected).abs()
            rel_error = error / expected.abs()

            rank = dist.get_rank()
            if rank == 0:
                print(f"tensor: {tensor[0][:8]} expected: {expected[0][:8]}")
                print(f"max diff: {error.max()} mean diff: {error.mean()} std diff: {error.std()}")
                print(
                    f"max rel diff: {rel_error.max()} mean rel diff: {rel_error.mean()} std rel diff: {rel_error.std()}"
                )

            assert rel_error.mean() < 0.1

    # Perform ring all-reduce
    processes = [mp.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("all_reduce_backend", [AllReduceBackend.GLOO, AllReduceBackend.CUSTOM])
def test_all_reduce_func(world_size, random_available_port, dist_environment, all_reduce_backend):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, "gloo", rank=rank, world_size=world_size):
            data = (rank + 1) * torch.ones(8, 8)
            ALL_REDUCE_FN[all_reduce_backend](
                data, op=dist.ReduceOp.SUM, group=dist.distributed_c10d._get_default_group()
            )

            assert data.mean() == sum([i + 1 for i in range(world_size)])

    processes = [mp.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
