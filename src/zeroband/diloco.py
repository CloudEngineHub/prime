from pydantic_config import BaseConfig
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch import nn
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int


def get_offloaded_param(model: nn.Module) -> list[torch.Tensor]:
    """
    Offload the model parameters to cpu
    """
    offloaded_params = []
    for param in model.parameters():
        if param.requires_grad:
            offloaded_param = param.data.detach().clone().to("cpu")
            offloaded_param.requires_grad = True
            offloaded_params.append(offloaded_param)

    return offloaded_params


class Diloco:
    def __init__(self, config: DilocoConfig, model: nn.Module, fsdp_sharding_strategy: ShardingStrategy):
        self.config = config
        self.fsdp_sharding_strategy = fsdp_sharding_strategy

        if self.fsdp_sharding_strategy != ShardingStrategy.FULL_SHARD:
            raise NotImplementedError("Only FULL_SHARD is supported for now")

        self._logger = get_logger()
        self.world_info = get_world_info()

        self._init_setup_device_mesh()
        self._init_offloaded_optimizer(model=model)

    def _init_setup_device_mesh(self):
        """Init two process group through device mesh, one local on gpu and one global on cpu"""
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

    def _init_offloaded_optimizer(self, model):
        self.cpu_model = get_offloaded_param(model)
        # todo: in case of sharded grap op we need to offload the cpu model only once per nodes

        self.outer_optimizer = torch.optim.SGD(self.cpu_model, lr=self.config.outer_lr, momentum=0.9, nesterov=True)

    def sync_pseudo_gradient(self, model: nn.Module):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """

        ### the whole sectione below is just a PoC. We need to benchmark and optimizer what is the most efficient:
        ## do the all reduce on cpu or on gpu
        ## do the outer optimizer step on cpu or on gpu

        ## right now we do all reduce on cpu

        for param_offloaded, param in zip(self.cpu_model, model.parameters()):
            # todo check how to handle the SHARD_GRAD_OP strategy where the weight are replicated across the local devices
            param_offloaded.grad = param_offloaded.data - param.data.to(param_offloaded.device)

            if param_offloaded.grad.device == torch.device("cpu"):
                # gloo does not support AVG
                param_offloaded.grad = param_offloaded.grad / self.global_pg.size()
                dist.all_reduce(param_offloaded.grad, op=dist.ReduceOp.SUM, group=self.global_pg)
            else:
                dist.all_reduce(param_offloaded.grad, op=dist.ReduceOp.AVG, group=self.global_pg)
