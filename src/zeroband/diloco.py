from pydantic_config import BaseConfig
import torch
from torch import nn
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int


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
        global_pg: dist.ProcessGroup,
    ):
        self.config = config
        self.fsdp_sharding_strategy = fsdp_sharding_strategy
        self.global_pg = global_pg

        self._logger = get_logger()
        self.world_info = get_world_info()

        if self.fsdp_sharding_strategy not in [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP]:
            raise ValueError("Diloco only support FULL_SHARD and SHARD_GRAD_OP")

        if self.world_info.global_world_size < 1:
            raise ValueError("Diloco requires a global world size of at least 1")

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
        self._logger.debug("sync pseudo gradient")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            if param.shape[0] == 0:
                continue
            param_offloaded.grad = param_offloaded.data - param.data.to(param_offloaded.device)

            # gloo does not support AVG
            param_offloaded.grad = param_offloaded.grad / self.global_pg.size()
            dist.all_reduce(param_offloaded.grad, op=dist.ReduceOp.SUM, group=self.global_pg)
            # todo async here

    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the CPU outer model to GPU
        """

        self._logger.debug("sync inner model")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param.data.copy_(param_offloaded.data)  # todo: use copy_ here

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
