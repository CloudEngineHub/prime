import torch
import torch.distributed as dist
from zeroband.C.collectives import ring_allreduce
from zeroband.collectives import ring_allreduce_py
from zeroband.C.compression import uniform_8bit_quantize
import math
import pytest
import multiprocessing as mp

N = 1_000_000
TIME_COUNT = 2


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("pg_source", ["gloo", "default"])
def test_ring_allreduce(world_size: int, pg_source: str, random_available_port: int, dist_environment):
    def all_reduce(rank: int, world_size: int):
        with dist_environment(random_available_port, "gloo", rank=rank, world_size=world_size):
            dist.init_process_group(backend="gloo")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if pg_source == "gloo":
                store = dist.TCPStore(
                    host_name="localhost",
                    port=random_available_port + 1,
                    world_size=world_size,
                    is_master=(rank == 0),
                )
                pg = dist.distributed_c10d.ProcessGroupGloo(store, rank, world_size)
            else:
                pg = dist.distributed_c10d._get_default_group()
            a = torch.randn(N) * 10
            b = torch.clone(a)
            c = torch.clone(a)

            ring_allreduce(a, dist.ReduceOp.SUM, pg)
            ring_allreduce_py(
                b,
                dist.ReduceOp.SUM,
                dist.distributed_c10d._get_default_group(),
                quantization_func=uniform_8bit_quantize,
            )
            dist.all_reduce(c, dist.ReduceOp.SUM, group=pg)

            if rank == 0:
                error_new = torch.norm(a - c)
                diff_new = (a - c).abs()
                error_old = torch.norm(b - c)
                diff_old = (b - c).abs()
                print(
                    f"[New] norm: {error_new:.4f} diff mean: {diff_new.mean():.4f} std: {diff_new.std()} max: {diff_new.max():.4f}"
                )
                print(
                    f"[Old] norm: {error_old:.4f} diff mean: {diff_old.mean():.4f} std: {diff_old.std()} max: {diff_old.max():.4f}"
                )

                assert (error_new - error_old).abs() / math.sqrt(N) < 0.5

            dist.destroy_process_group()

    # Perform ring all-reduce
    processes = [mp.Process(target=all_reduce, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
