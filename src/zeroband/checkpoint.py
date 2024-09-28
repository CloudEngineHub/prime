from dataclasses import dataclass
import time
from typing import Any
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed.checkpoint as dcp
from torch.distributed import ProcessGroup
from torch.distributed.checkpoint.state_dict import (
    set_optimizer_state_dict,
    set_model_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from zeroband.utils.logging import get_logger
import warnings
import logging

## code inspired by torchtitan https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py


GLOBAL_STATE_FILE = "global_state_dict.pt"


@dataclass
class TrainingProgress(Stateful):
    total_tokens: int
    outer_step: int
    step: int

    def state_dict(self) -> dict[str, Any]:
        return {"total_tokens": self.total_tokens, "outer_step": self.outer_step, "step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_tokens = state_dict["total_tokens"]
        self.outer_step = state_dict["outer_step"]
        self.step = state_dict["step"]


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(model=self.model, model_state_dict=state_dict, options=StateDictOptions(strict=False))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.optim = optim

    def state_dict(self) -> dict[str, Any]:
        return get_optimizer_state_dict(
            model=self.model, optimizers=self.optim, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_optimizer_state_dict(
            model=self.model, optimizers=self.optim, optim_state_dict=state_dict, options=StateDictOptions(strict=False)
        )


class CkptManager:
    """Its name CkptManager because I (sami) always misstyped chekcpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: StatefulDataLoader,
        training_progress: TrainingProgress,
        process_group: ProcessGroup | None,
    ):
        self.model = ModelWrapper(model)
        self.optimizer = OptimizerWrapper(model, optimizer)
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.training_progress = training_progress

        # states can only be stateful object, hence we need to wrap Model and Optimizer
        self.states: dict[str, Stateful] = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "dataloader": self.dataloader,
            "training_progress": self.training_progress,
        }

        self.process_group = process_group
        self._logger = get_logger()

    def save(self, ckpt_path: str) -> None:
        """
        Each rank will save the right shard of the model and optimizer.

        Saving is done inplace
        """

        time_start = time.perf_counter()

        catch_warning = self._logger.getEffectiveLevel() <= logging.INFO
        # pytorch has an annoying warning when saving the optimizer state https://github.com/pytorch/pytorch/issues/136907
        # we can ignore it if we are not logging in DEBUG mode

        with warnings.catch_warnings():
            if catch_warning:
                warnings.simplefilter("ignore")

            dcp.save(self.states, process_group=self.process_group, checkpoint_id=ckpt_path)

        self._logger.info(f"Saved checkpoint to {ckpt_path} in {time.perf_counter() - time_start} seconds")

    def load(self, resume_ckpt_path: str) -> None:
        """
        loading should be done after fsdp wrap and optimizer init.
        Each rank will load the right shard of the model and optimizer.
        All rank will load the global states (scheduler, step, total_tokens, dataloader).

        Loading is done inplace
        """
        time_start = time.perf_counter()
        self.states = dcp.load(self.states, process_group=self.process_group, checkpoint_id=resume_ckpt_path)
        self._logger.info(f"Loaded checkpoint from {resume_ckpt_path} in {time.perf_counter() - time_start} seconds")


# def save(
#     checkpoint_path: str,
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LambdaLR,
#     outer_optimizer: torch.optim.Optimizer | None = None,
#     scaler: torch.cuda.amp.GradScaler | None = None,
#     loss: float | None = None,
#     data_loader: StatefulDataLoader | None = None,
#     save_global_state: bool = True,
# ):
#     """Save the model and optimizer state to a checkpoint folderx

#     Args:
#         checkpoint_path: the path to the checkpoint folder
#         model: the model to save
#         optimizer: the optimizer to save
#         scheduler: the scheduler to save
#         outer_optimizer: the outer optimizer to save
#         loss: the loss to save
#         data_loader: the data loader to save
#         save_global_state: whether to save the global state
#     """
#     rank = int(os.environ["RANK"])

#     # 1. Save distributed states
#     # fs_storage_writer = dcp.FsspecWriter(checkpoint_path, sync_files=False)
#     # for some reason sync_files = True try to call stream.fileno which is not supported with gcp ffspec storage.

#     model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
#     dcp_state_dict = {
#         "model": model_state_dict,
#         "optimizer": optimizer_state_dict,
#     }
#     dcp.save(dcp_state_dict, checkpoint_id=checkpoint_path)
#     if data_loader is not None:
#         rank_state_dict = {}
#         rank_state_dict["data_loader"] = data_loader.state_dict()
#         with open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "wb") as f:
#             torch.save(rank_state_dict, f)

#     if not save_global_state:
#         return

#     # 2. Save global states
#     global_state_dict = {"scheduler": scheduler.state_dict(), "loss": loss if loss is not None else 0}
#     if outer_optimizer is not None:
#         global_state_dict["outer_optimizer"] = outer_optimizer.state_dict()
#     if scaler is not None:
#         global_state_dict["scaler"] = scaler.state_dict()

#     with open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "wb") as f:
#         torch.save(global_state_dict, f)

# def load_checkpoint(
#     checkpoint_path: str,
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
#     outer_optimizer: torch.optim.Optimizer | None = None,
#     scaler: torch.cuda.amp.GradScaler | None = None,
#     data_loader: StatefulDataLoader | None = None,
# ) -> float:
#     """Load the model and optimizer state from a checkpoint folder

#     Args:
#         checkpoint_path: the path to the checkpoint folder
#         model: the model to load
#         optimizer: the optimizer to load
#         scheduler: the scheduler to load
#         outer_optimizer: the outer optimizer to load
#         data_loader: the data loader to load

#     Returns:
#         loss: the loss from the checkpoint
#     """
#     rank = int(os.environ["RANK"])
#     # 1. Load distributed states
#     # fs_storage_reader = dcp.FsspecReader(checkpoint_path)

#     model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
#     dcp_state_dict = {
#         "model": model_state_dict,
#         "optimizer": optimizer_state_dict,
#     }
#     dcp.load(dcp_state_dict, checkpoint_id=checkpoint_path)
#     set_state_dict(
#         model,
#         optimizer,
#         model_state_dict=model_state_dict,
#         optim_state_dict=optimizer_state_dict,
#     )
#     if data_loader is not None:
#         with open(os.path.join(checkpoint_path, f"__{rank}_0.pt"), "rb") as f:
#             rank_state_dict = torch.load(f)
#         data_loader.load_state_dict(rank_state_dict["data_loader"])

#     # 2. Load global states
#     with open(os.path.join(checkpoint_path, GLOBAL_STATE_FILE), "rb") as f:
#         global_state_dict = torch.load(f)
#     if scheduler is not None:
#         scheduler.load_state_dict(global_state_dict["scheduler"])
#         optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
#     if outer_optimizer is not None:
#         outer_optimizer.load_state_dict(global_state_dict["outer_optimizer"])
#     if scaler is not None:
#         scaler.load_state_dict(global_state_dict["scaler"])
#     return global_state_dict["loss"]


# class CkptManager:
#     """Its name CkptManager because I (sami) always misstyped chekcpoint. """

#     def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, dataloader: StatefulDataLoader, training_progress: TrainingProgress, process_group: ProcessGroup | None):

#         self.model = model
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.dataloader = dataloader
#         self.training_progress = training_progress

#         # states can only be stateful object, hence we need to wrap Model and Optimizer

#         self.process_group = process_group
#         self._logger = get_logger()

#     def save(self, ckpt_path: str) -> None:
#         save(checkpoint_path=ckpt_path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, data_loader=self.dataloader)


#     def load(self, resume_ckpt_path: str) -> None:
#         load_checkpoint(checkpoint_path=resume_ckpt_path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, data_loader=self.dataloader)
