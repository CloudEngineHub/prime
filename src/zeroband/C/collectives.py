import os
from typing import Optional
import torch
import torch.distributed as dist
from torch.utils import cpp_extension
from pathlib import Path
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup


parent = Path(__file__).parent
INCLUDES = [str(parent / "csrc"), str(parent.parent.parent.parent / "third_party/gloo")]
COLLECTIVES_CSRC_PATH = parent / "csrc" / "collectives.cpp"

collectives_ops = cpp_extension.load(
    name="collectives",
    sources=[COLLECTIVES_CSRC_PATH],
    extra_cflags=["-O3", "-DUSE_C10D_GLOO"],
    verbose=False if os.environ.get("ZERO_BAND_LOG_LEVEL") == "DEBUG" else True,
    extra_include_paths=INCLUDES,
)


def ring_allreduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if isinstance(group, dist.distributed_c10d.ProcessGroupGloo):
        collectives_ops.ring_allreduce_gloo(tensor, op, group)
    elif isinstance(group, FakeProcessGroup):
        return
    else:
        collectives_ops.ring_allreduce(tensor, op, group)
