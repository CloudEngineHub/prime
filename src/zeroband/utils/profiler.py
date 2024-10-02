import os
import pickle
import torch
from zeroband.utils.logging import get_logger

from zeroband.utils.world_info import get_world_info


_MAX_ENTRIES = 10000


class MemoryProfiler:
    """Pytorch Memory Profiler.
    The output are pickles file that can be visualized here: https://pytorch.org/memory_viz
    """

    def __init__(self, freq: int, snapshot_dir: str):
        torch.cuda.memory._record_memory_history(max_entries=_MAX_ENTRIES)
        self.freq = freq

        self.world_info = get_world_info()
        self.logger = get_logger()
        self.step_num = 0

        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshot_dir = snapshot_dir

    def step(self):
        self.step_num += 1
        if self.step_num % self.freq != 0:
            return

        dir_name = f"iteration_{self.step_num}"

        curr_snapshot_dir = os.path.join(self.snapshot_dir, dir_name)
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)

        with open(f"{curr_snapshot_dir}/rank{self.world_info.rank}_memory_snapshot.pickle", "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)

        torch.distributed.barrier()
