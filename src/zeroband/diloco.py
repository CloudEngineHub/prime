import os
import shutil
from pydantic_config import BaseConfig
import torch
from torch import nn
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from zeroband.comms import ElasticDeviceMesh
from torch.distributed.fsdp import ShardingStrategy
import torch.distributed as dist
from zeroband.testing import get_module_signature
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


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

    def _init_offloaded_optimizer(self, model: nn.Module):
        with FSDP.summon_full_params(model):
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
        works = []
        # TODO: This assumes all params require grad, which is used by the offload
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param_offloaded.grad = param_offloaded.data - param.data.to(param_offloaded.device)

            # gloo does not support AVG
            param_offloaded.grad = param_offloaded.grad / self.elastic_device_mesh.global_pg.size()
            work = dist.all_reduce(
                param_offloaded.grad, op=dist.ReduceOp.SUM, group=self.elastic_device_mesh.global_pg, async_op=True
            )
            works.append(work)
        for work in works:
            work.wait()

    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the CPU outer model to GPU
        """

        self._logger.debug("sync inner model")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param.data.copy_(param_offloaded.data)

    def get_offloaded_param(self, model: nn.Module) -> list[nn.Parameter]:
        """
        Offload the model parameters to cpu
        """
        offloaded_params = []
        os.makedirs(f"/dev/shm/zeroband/{self.world_info.global_unique_id}", exist_ok=True)

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                storage = torch.UntypedStorage.from_file(
                    f"/dev/shm/zeroband/{self.world_info.global_unique_id}/{param_name}",
                    True,
                    param.data.untyped_storage().size(),
                )
                offloaded_param = torch.tensor(storage, dtype=param.dtype, device="cpu")
                offloaded_param.as_strided_(size=param.data.size(), stride=param.data.stride())
                if self.world_info.rank == 0:
                    # TODO: Can we async or split the copy among gpus probs overkill?
                    offloaded_param.copy_(param.data)
                offloaded_param.requires_grad = False  # TODO: check if we need to set this to True
                offloaded_params.append(offloaded_param)

        dist.barrier()
        return offloaded_params

    def step(self, model: nn.Module):
        """
        Step the optimizer
        """
        with FSDP.summon_full_params(model):
            self._logger.debug("Pre diloco step %s", get_module_signature(model))
            if self.world_info.rank == 0:
                self.sync_pseudo_gradient(model)
                if self.outer_optimizer is not None:
                    self.outer_optimizer.step()
                    self.outer_optimizer.zero_grad()  # todo(sami): check if we can remove this

            dist.barrier()
            self.sync_inner_model(model)
            self._logger.debug("Post meow diloco step %s", get_module_signature(model))

    def __del__(self):
        shutil.rmtree(f"/dev/shm/zeroband/{self.world_info.global_unique_id}", ignore_errors=True)
