import time
from pydantic_config import BaseConfig
import torch
from torch import nn
from zeroband.collectives import Compression, all_reduce
from zeroband.comms import ElasticDeviceMesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor


class DilocoConfig(BaseConfig):
    outer_lr: float = 0.7
    inner_steps: int
    compression: Compression = Compression.NO

    retry_all_reduce: int = 3


class Diloco:
    """
    This class implements the diloco algorithm from  https://arxiv.org/abs/2311.08105 and https://arxiv.org/abs/2407.07852.

    It handles the outer loop as well as the inter node communication.

    There is no VRAM overhead with this implementation as the model is outer optimizer is offloaded to cpu.
    All reduce communication are also done on cpu using GLOO.

    Example usage:

    # Example usage in a training loop:

    diloco = Diloco(config.diloco, model, elastic_device_mesh)

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
        elastic_device_mesh: ElasticDeviceMesh,
    ):
        self.config = config

        if config.compression == Compression.UINT8:
            from zeroband.C.collectives import ring_allreduce as _  # noqa: F401
            # just force compilation

        self.elastic_device_mesh = elastic_device_mesh

        self._logger = get_logger()
        self.world_info = get_world_info()

        self._init_offloaded_optimizer(model=model)

    def _init_offloaded_optimizer(self, model):
        self.param_list_cpu = self.get_offloaded_param(model)
        self.outer_optimizer = torch.optim.SGD(
            self.param_list_cpu, lr=self.config.outer_lr, momentum=0.9, nesterov=True
        )
        self._logger.debug("offload model to cpu")

    def sync_pseudo_gradient(self, model: nn.Module, fake: bool = False):
        """
        Sync the pseudo gradient from the local process group to the global process group
        """
        self._logger.debug("sync pseudo gradient" + " fake" if fake else "")

        self.elastic_device_mesh.maybe_reinit_global_pg(admit_joiners=False)
        global_pg = self.elastic_device_mesh.global_pg
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            if param.shape[0] == 0:
                continue

            for i in range(self.config.retry_all_reduce):
                try:
                    if fake:
                        grad = torch.zeros_like(param_offloaded.data.to_local())
                    else:
                        grad = param_offloaded.data.to_local() - param.data.to_local().to(param_offloaded.data.device)

                    grad = grad / global_pg.size()
                    all_reduce(self.config.compression, grad, dist.ReduceOp.SUM, global_pg)
                    # self._logger.debug(f"all_reduce {i} done")
                    break
                except RuntimeError as e:
                    self._logger.error(
                        f"Error syncing pseudo gradient: {e}, retry {i+1}/{self.config.retry_all_reduce}"
                    )
                    global_pg = self.elastic_device_mesh.get_global_pg(maybe_reinit=True)

            param_offloaded.grad.to_local().copy_(grad)

            # todo async here

    def sync_inner_model(self, model: nn.Module):
        """
        Sync the inner model from the CPU outer model to GPU
        """

        self._logger.debug("sync inner model")
        for param_offloaded, param in zip(self.param_list_cpu, model.parameters()):
            param.data.to_local().copy_(param_offloaded.data.to_local())

    def get_offloaded_param(self, model: nn.Module) -> list[nn.Parameter]:
        """
        Offload the model parameters to cpu
        """
        offloaded_params = []

        for param in model.parameters():
            if param.requires_grad:
                # so here we copy the DTensor from gpu to cpu. The trick is that we need to recreate the DTensor with the correct
                # cpu devise mesh, otherwise we have a cpu DTensor with a cuda device mesh which will fail to do any communication

                offloaded_param = nn.Parameter(
                    DTensor.from_local(
                        param.data.to_local().detach().to("cpu"),
                        device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                        placements=param.data.placements,
                    )
                )
                offloaded_param.grad = DTensor.from_local(
                    torch.zeros_like(param.data.to_local()),
                    device_mesh=self.elastic_device_mesh.cpu_local_mesh,
                    placements=param.data.placements,
                )
                # here we pre-allocate the grad DTensor on cpu.
                offloaded_param.requires_grad = True
                offloaded_params.append(offloaded_param)

        return offloaded_params

    def step(self, model: nn.Module, fake: bool = False):
        """
        Step the optimizer
        """
        time_start = time.perf_counter()
        self.sync_pseudo_gradient(model, fake=fake)
        self._logger.info(f"all reduce pseudo gradient in: {time.perf_counter() - time_start} seconds")

        if self.outer_optimizer is not None:
            self.outer_optimizer.step()

        self.sync_inner_model(model)
