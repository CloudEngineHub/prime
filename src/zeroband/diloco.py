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


class ElasticDeviceMesh:
    """Init two process group through device mesh, one local on gpu and one global on cpu"""

    def __init__(self):
        self._logger = get_logger()

        self.world_info = get_world_info()

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


class Diloco:
    def __init__(
        self,
        config: DilocoConfig,
        model: nn.Module,
        fsdp_sharding_strategy: ShardingStrategy,
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.config = config
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.elastic_device_mesh = elastic_device_mesh

        self._logger = get_logger()
        self.world_info = get_world_info()

        self.need_to_offload = (
            self.fsdp_sharding_strategy == ShardingStrategy.FULL_SHARD or self.world_info.local_rank == 0
        )
        # if we are not in fully sharded mode only the local rank 0 will have the model on cpu
        if self.need_to_offload:
            self._init_offloaded_optimizer(model=model)
        else:
            self.outer_optimizer = None
            self.cpu_model = None

    def _init_offloaded_optimizer(self, model):
        self.cpu_model = self.get_offloaded_param(model)
        self.outer_optimizer = torch.optim.SGD(self.cpu_model, lr=self.config.outer_lr, momentum=0.9, nesterov=True)
        self._logger.debug("offload model to cpu")

    def sync_pseudo_gradient(self, model: nn.Module):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """
        if self.need_to_offload:
            self._logger.debug("sync pseudo gradient")
            for param_offloaded, param in zip(self.cpu_model, model.parameters()):
                # todo check how to handle the SHARD_GRAD_OP strategy where the weight are replicated across the local devices
                param_offloaded.grad = param_offloaded.data - param.data.to(param_offloaded.device)

                # gloo does not support AVG
                param_offloaded.grad = param_offloaded.grad / self.elastic_device_mesh.global_pg.size()
                dist.all_reduce(param_offloaded.grad, op=dist.ReduceOp.SUM, group=self.elastic_device_mesh.global_pg)

    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the global process group to the local process group
        """

        if self.fsdp_sharding_strategy == ShardingStrategy.FULL_SHARD:
            # here each rank has a shard of the model in memory so all rank do the sync
            self._logger.debug("sync inner model")
            for param_offloaded, param in zip(self.cpu_model, model.parameters()):
                param.data = param_offloaded.data.to("cuda")

        elif self.fsdp_sharding_strategy in [ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]:
            self._logger.debug("sync inner model")
            # in shard_grad_op mode, only the local rank 0 has the model in cpu
            # we first copy the model to the gpu 0 and then broadcast it to the other gpu as
            # gpu to gpu is faster than cpu to gpu with nvlink

            for i, (param_offloaded, param) in enumerate(zip(self.cpu_model, model.parameters())):
                # todo: we can probably overlap both comm here
                if self.world_info.local_rank == 0:
                    self._logger.debug(
                        f"i: {i}  shape param {param.data.shape} shape offloaded {param_offloaded.data.shape}"
                    )
                    param.data = param_offloaded.data.to("cuda")

                dist.broadcast(tensor=param.data, src=0, group=self.elastic_device_mesh.local_pg)

    def get_offloaded_param(self, model: nn.Module) -> list[nn.Parameter]:
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

    def step(self, model: nn.Module):
        """
        Step the optimizer
        """
        # self.sync_pseudo_gradient(model)
        # if self.outer_optimizer is not None:
        #     self.outer_optimizer.step()
        #     self.outer_optimizer.zero_grad()  # todo(sami): check if we can remove this

        for param in model.parameters():
            param.data = torch.zeros_like(param.data).to(param.data.device)

        # self.sync_inner_model(model)
