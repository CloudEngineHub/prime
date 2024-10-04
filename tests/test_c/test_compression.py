import torch
from torch.utils.benchmark import Timer
from zeroband.compression import uniform_8bit_quantize as uniform_8bit_quantize_old
from zeroband.compression import average_buckets as average_buckets_old

from zeroband.C.compression import average_buckets, uniform_8bit_quantize, quantize_per_tensor_uint8

N = 10_000_000
TIME_COUNT = 1


def test_uniform_8bit_quantize():
    a = torch.randn(N)

    # Benchmark old function
    timer_old = Timer(
        stmt="uniform_8bit_quantize_old(a)", globals={"uniform_8bit_quantize_old": uniform_8bit_quantize_old, "a": a}
    )
    time_old = timer_old.timeit(TIME_COUNT)

    # Benchmark new function
    timer_new = Timer(stmt="uniform_8bit_quantize(a)", globals={"uniform_8bit_quantize": uniform_8bit_quantize, "a": a})
    time_new = timer_new.timeit(TIME_COUNT)

    print(f"New function time: {time_new.mean:.6f} seconds")
    print(f"Old function time: {time_old.mean:.6f} seconds")

    new_result, new_lookup = uniform_8bit_quantize(a)
    old_result, old_lookup = uniform_8bit_quantize_old(a)

    new_ten = new_lookup[new_result.long()]
    old_ten = old_lookup[old_result.long()]

    new_err = torch.norm(new_ten - a)
    old_err = torch.norm(old_ten - a)
    new_diff = (new_ten - a).abs()
    old_diff = (old_ten - a).abs()
    print(
        f"New error: {new_err:.6f} Diff mean: {new_diff.mean():.6f} Std: {new_diff.std():.6f} Max: {new_diff.max():.6f}"
    )
    print(
        f"Old error: {old_err:.6f} Diff mean: {old_diff.mean():.6f} Std: {old_diff.std():.6f} Max: {old_diff.max():.6f}"
    )


def test_quantize_per_tensor_uint8():
    a = torch.ones(N) * 10
    scale = 0.01
    print(f"Tensor size: {a.numel():,}")

    timer_new = Timer(
        stmt="quantize_per_tensor(a, scale, 128)",
        globals={"quantize_per_tensor": quantize_per_tensor_uint8, "a": a, "scale": scale},
    )
    time_new = timer_new.timeit(TIME_COUNT)
    print(f"Custom quantize_per_tensor function time: {time_new.mean:.6f} seconds")

    timer_old = Timer(
        stmt="torch.quantize_per_tensor(a, scale, 128, torch.quint8).int_repr()",
        globals={"torch": torch, "a": a, "scale": scale},
    )
    time_old = timer_old.timeit(TIME_COUNT)
    print(f"torch.quantize_per_tensor time: {time_old.mean:.6f} seconds")


def test_average_buckets():
    a = torch.randn(N) * 10
    b = torch.randint(0, 255, (N,), dtype=torch.uint8)

    timer_new = Timer(stmt="average_buckets(a, b, 256)", globals={"average_buckets": average_buckets, "a": a, "b": b})
    time_new = timer_new.timeit(TIME_COUNT)
    print(f"Custom average_buckets function time: {time_new.mean:.6f} seconds")

    timer_old = Timer(
        stmt="average_buckets(a, b, 256)", globals={"average_buckets": average_buckets_old, "a": a, "b": b}
    )
    time_old = timer_old.timeit(TIME_COUNT)
    print(f"torch.bucketize time: {time_old.mean:.6f} seconds")
