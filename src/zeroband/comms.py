from torch.distributed.device_mesh import init_device_mesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
import torch.distributed as dist


class ElasticDeviceMesh:
    """Init two process group through device mesh, one local on gpu and one global on cpu"""

    def __init__(self):
        self._logger = get_logger()

        self.world_info = get_world_info()

        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        # right now device mesh does not support two backend so we just create two identicaly mesh expect the backend
        self.device_mesh = init_device_mesh(
            "cuda", (self.world_info.nnodes, self.world_info.local_world_size), mesh_dim_names=("global", "local")
        )
        self.device_mesh_cpu = init_device_mesh(
            "gloo", (self.world_info.nnodes, self.world_info.local_world_size), mesh_dim_names=("global", "local")
        )

        self.global_pg = self.device_mesh_cpu.get_group("global")
        self.local_pg = self.device_mesh.get_group("local")

        self._logger.debug(f"global pg world : {self.global_pg.size()}, local pg: {self.local_pg.size()}")
    
    def __del__(self):
        dist.destroy_process_group()
