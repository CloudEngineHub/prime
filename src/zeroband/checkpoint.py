from dataclasses import dataclass
import gc
import multiprocessing
import os
import time
from typing import Any
from fsspec.generic import rsync as rsync_fsspec
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed.checkpoint as dcp
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

from zeroband.utils.world_info import get_world_info

## code inspired by torchtitan https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py


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
    """Its name CkptManager because I (sami) always misstyped chekcpoint.

    Checkpoint are saved in a folder with the following structure:
    ckpt_path/
        step_0/
            _0_0.pt
            _1_0.pt
            ...
        step_1/
            ...
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: StatefulDataLoader,
        training_progress: TrainingProgress,
        diloco_offloaded_param_list: list[nn.Parameter] | None,
        diloco_offloaded_optimizer: Optimizer | None,
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
            # "dataloader": self.dataloader, # ignoring dataloader for now as each rank has its own dataloader
            "training_progress": self.training_progress,
        }

        assert (diloco_offloaded_param_list is None) == (
            diloco_offloaded_optimizer is None
        ), "diloco_offloaded_model and diloco_offloaded_optimizer must be both None or both have values"

        self.diloco_offloaded_optimizer = diloco_offloaded_optimizer  # he we don't use Wrapper because it failed
        # which might make the ckpt less generic in term of loading from different number of device. FSDP ckpt seems to be a mess tho
        self.diloco_offloaded_param_list = diloco_offloaded_param_list

        if diloco_offloaded_optimizer is not None:
            # even if the diloco_offloaded target the cpu list model, we still use the gpu model to load and save state.
            # main reason is that we actually don't a cpu model but just a list of cpu parameters.
            self.states["diloco_optimizer"] = self.diloco_offloaded_optimizer

        self._logger = get_logger()

        self.async_save_process: list[multiprocessing.Process] = []

    def save(self, ckpt_path: str, remote_ckpt_path: str | None) -> None:
        """
        Each rank will save the right shard of the model and optimizer.

        Saving is done inplace
        """

        time_start = time.perf_counter()
        world_info = get_world_info()

        ckpt_path = os.path.join(ckpt_path, f"step_{self.training_progress.step}")
        if self.diloco_offloaded_optimizer:
            # here we save model and offloaded optimizer on each diloco rank even tho they are the same
            # this is done for two reasons:
            #   * if the nodes don't share a filesystem nor a remote path, they still save all of the data
            #   * its easier to implement and avoid race condition on the shared data.
            ckpt_path = os.path.join(ckpt_path, f"diloco_{world_info.diloco_rank}")

        catch_warning = self._logger.getEffectiveLevel() <= logging.INFO

        with warnings.catch_warnings():
            # pytorch has an annoying warning when saving the optimizer state https://github.com/pytorch/pytorch/issues/136907
            # we can ignore it if we are not logging in DEBUG mode
            if catch_warning:
                warnings.simplefilter("ignore")

            dcp.save(self.states, checkpoint_id=ckpt_path)

            ## the next part is a fix so that each rank save a different dataloader rank. It not efficient because it reads the state two times from disk
            with open(os.path.join(ckpt_path, f"__{world_info.local_rank}_0.pt"), "wb") as f:
                torch.save({"data_loader": self.dataloader.state_dict()}, f)

        self._logger.info(f"Saved checkpoint to {ckpt_path} in {time.perf_counter() - time_start} seconds")

        gc.collect()  # because we are badass engineer

        if remote_ckpt_path is not None:
            self._async_save_remote(ckpt_path, remote_ckpt_path)

    def _async_save_remote(self, ckpt_path: str, remote_ckpt_path: str):
        """asyncronously rsync a ckpt folder to a remote location. Using fsspec to handle remote cloud storage without to install
        specific libraries (e.g. s3fs)
        """

        def rsync():
            time_start = time.perf_counter()
            self._logger.info(f"start pushing {ckpt_path} to {remote_ckpt_path} asynchronously")
            rsync_fsspec(ckpt_path, destination=remote_ckpt_path)
            self._logger.info(
                f"finish pushing {ckpt_path} to {remote_ckpt_path} in {time.perf_counter() - time_start} seconds"
            )

        processes = multiprocessing.Process(target=rsync, daemon=True)
        processes.start()

        self.async_save_process.append(processes)

    def wait_async_save_process(self):
        """
        wait for all async save process to finish
        """
        for process in self.async_save_process:
            process.join()

    def _del__(self):
        self.wait_async_save_process()

    def load(self, resume_ckpt_path: str) -> None:
        """
        loading should be done after fsdp wrap and optimizer init.
        Each rank will load the right shard of the model and optimizer.
        All rank will load the global states (scheduler, step, total_tokens, dataloader).

        `resume_ckpt_path` should point to a specific step and not to the base ckpt folder. Example: `ckpt_path/step_100`

        Loading is done inplace
        """
        time_start = time.perf_counter()

        world_info = get_world_info()
        if self.diloco_offloaded_param_list is not None:
            resume_ckpt_path = os.path.join(resume_ckpt_path, f"diloco_{world_info.diloco_rank}")

        self.states = dcp.load(self.states, checkpoint_id=resume_ckpt_path)

        # since we don't load the param list from the state dict as its the same as the model one we just copy
        if self.diloco_offloaded_param_list is not None:
            for param_offloaded, param_model in zip(self.diloco_offloaded_param_list, self.model.model.parameters()):
                param_offloaded.data.copy_(param_model.data)

        ## the next part is a fix so that each rank save a different dataloader rank. It not efficient because it reads the state two times from disk
        with open(os.path.join(resume_ckpt_path, f"__{world_info.local_rank}_0.pt"), "rb") as f:
            rank_state_dict = torch.load(f)

        self.dataloader.load_state_dict(rank_state_dict["data_loader"])

        self._logger.info(f"Loaded checkpoint from {resume_ckpt_path} in {time.perf_counter() - time_start} seconds")
