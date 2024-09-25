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
    """
    This class implements the diloco algorithm from  https://arxiv.org/abs/2311.08105 and https://arxiv.org/abs/2407.07852.

    It handles the outer loop as well as the inter node communication.

    There is no VRAM overhead with this implementation as the model is outer optimizer is offloaded to cpu.
    All reduce communication are also done on cpu using GLOO.

    Example usage:

    # Example usage in a training loop:

    diloco = Diloco(config.diloco, model, sharding_strategy, elastic_device_mesh)

    for outer_step in range(num_outer_steps):
        for inner_step in range(config.diloco.inner_steps):
            # Regular inner training loop
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()

        diloco.step(model)
    """

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

        if self.fsdp_sharding_strategy not in [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP]:
            raise ValueError("Diloco only support FULL_SHARD and SHARD_GRAD_OP")

        self._init_offloaded_optimizer(model=model)

    def _init_offloaded_optimizer(self, model):
        self.param_list_cpu = self.get_offloaded_param(model)
        self.outer_optimizer = torch.optim.SGD(
            self.param_list_cpu, lr=self.config.outer_lr, momentum=0.9, nesterov=True
        )
        self._logger.debug("offload model to cpu")

    def sync_pseudo_gradient(self, model: nn.Module):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """
        if self.need_to_offload:
            self._logger.debug("sync pseudo gradient")
            for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
                # todo check how to handle the SHARD_GRAD_OP strategy where the weight are replicated across the local devices
                param_offloaded.grad = param_offloaded.data - param.data.to(param_offloaded.device)

                # gloo does not support AVG
                param_offloaded.grad = param_offloaded.grad / self.elastic_device_mesh.global_pg.size()
                dist.all_reduce(
                    param_offloaded.grad, op=dist.ReduceOp.SUM, group=self.elastic_device_mesh.global_pg, async_op=True
                )
                # todo async here

    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the global process group to the local process group
        """

        self._logger.debug("sync inner model")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param.data = param_offloaded.data.to("cuda")  # todo: use copy_ here

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
        self.sync_pseudo_gradient(model)
        if self.outer_optimizer is not None:
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()  # todo(sami): check if we can remove this

        self.sync_inner_model(model)
