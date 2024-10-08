from dataclasses import dataclass
import gc
import multiprocessing
import os
import shutil
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
import torch.distributed as dist


from torch.distributed.checkpoint.stateful import Stateful
from zeroband.utils.logging import get_logger
import warnings
import logging
from zeroband.utils.wget import wget
from torch.distributed._tensor.api import DTensor

from zeroband.utils.world_info import get_world_info

## code inspired by torchtitan https://github.com/pytorch/torchtitan/blob/main/torchtitan/checkpoint.py

SHM_PATH = "/dev/shm/zeroband"


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
        return get_model_state_dict(self.model, options=StateDictOptions(strict=False))

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
            model=self.model,
            optimizers=self.optim,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


def cast_dtensor_to_tensor(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Traverse a state dict and cast all DTensor in the state dict to tensor
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if isinstance(value, dict):
            new_state_dict[key] = cast_dtensor_to_tensor(value)
        elif isinstance(value, DTensor):
            new_state_dict[key] = value.to_local()
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_dtensor_state_dict(state_src, loaded_state_dict):
    for key, value in state_src.items():
        if isinstance(value, dict):
            load_dtensor_state_dict(value, loaded_state_dict[key])
        elif isinstance(value, DTensor):
            local_tensor = value.to_local()

            local_tensor.copy_(loaded_state_dict[key])
            loaded_state_dict[key] = value
        else:
            loaded_state_dict[key] = value


class OuterOptimizerWrapper(Stateful):
    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        # the idea here is to cast any DTensor into local tensor
        state = self.optimizer.state_dict()
        return cast_dtensor_to_tensor(state)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # we pre-init the opt buffer DTensor.
        # !! this assume that the model have grad buffer init
        self.optimizer.step()  # pre init buffer

        ## here the idea is for any DTensor, load the value from the state_dict into the local tensor
        current_state = self.optimizer.state_dict()
        load_dtensor_state_dict(current_state, state_dict)
        self.optimizer.load_state_dict(state_dict)


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

    states: dict[str, Stateful]

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: StatefulDataLoader,
        training_progress: TrainingProgress,
        diloco_offloaded_param_list: list[nn.Parameter] | None,
        diloco_offloaded_optimizer: Optimizer | None,
        live_ckpt_server: bool = False,
        live_recovery_port: int | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.training_progress = training_progress

        assert (diloco_offloaded_param_list is None) == (
            diloco_offloaded_optimizer is None
        ), "diloco_offloaded_model and diloco_offloaded_optimizer must be both None or both have values"

        self.diloco_offloaded_optimizer = diloco_offloaded_optimizer  # he we don't use Wrapper because it failed
        # which might make the ckpt less generic in term of loading from different number of device. FSDP ckpt seems to be a mess tho
        self.diloco_offloaded_param_list = diloco_offloaded_param_list

        self._init_state()

        self._logger = get_logger()
        self.world_info = get_world_info()

        self.async_save_process: list[multiprocessing.Process] = []

        if live_ckpt_server:
            self.shm_path = os.path.join(SHM_PATH, self.world_info.global_unique_id, "latest")
            shutil.rmtree(self.shm_path, ignore_errors=True)
            os.makedirs(self.shm_path, exist_ok=True)
            self.live_server = CkptLiveServer(port=live_recovery_port, ckpt_path=self.shm_path)
        else:
            self.shm_path = None

    def _init_state(self):
        # states can only be stateful object, hence we need to wrap Model and Optimizer
        self.states: dict[str, Stateful] = {
            "model": ModelWrapper(self.model),
            "optimizer": OptimizerWrapper(self.model, self.optimizer),
            "scheduler": self.scheduler,
            # "dataloader": self.dataloader, # ignoring dataloader for now as each rank has its own dataloader
            "training_progress": self.training_progress,
        }

        # if self.diloco_offloaded_optimizer is not None:
        #     # even if the diloco_offloaded target the cpu list model, we still use the gpu model to load and save state.
        #     # main reason is that we actually don't a cpu model but just a list of cpu parameters.
        #     self.states["diloco_optimizer"] = self.diloco_offloaded_optimizer

    def save_shm(self) -> None:
        """
        Save the latest checkpoint in shared memory.
        """
        time_start = time.perf_counter()
        ckpt_path = self.shm_path
        self._save(ckpt_path)
        self._logger.info(f"Saved checkpoint to {ckpt_path} in {time.perf_counter() - time_start} seconds")

    def save(self, ckpt_path: str, remote_ckpt_path: str | None, already_in_shm: bool = False) -> None:
        """
        Each rank will save the right shard of the model and optimizer.

        Saving is done inplace.

        Save in the subfolder `step_<step>`.

        shm_save=True mean we previsouly saved to shm so we just do a copy past to disk
        """

        time_start = time.perf_counter()

        step_ckpt_path = os.path.join(ckpt_path, f"step_{self.training_progress.step}")

        if not already_in_shm:
            self._save(ckpt_path)
            if remote_ckpt_path is not None:
                self._async_save_remote(step_ckpt_path, remote_ckpt_path)
            self._logger.info(f"Saved checkpoint to {ckpt_path} in {time.perf_counter() - time_start} seconds")

        else:
            self._async_save_remote(
                self.shm_path, step_ckpt_path, os.path.join(remote_ckpt_path, f"step_{self.training_progress.step}")
            )

    def _save(self, ckpt_path: str):
        if self.diloco_offloaded_optimizer:
            # here we save model and offloaded optimizer on each diloco rank even tho they are the same
            # this is done for two reasons:
            #   * if the nodes don't share a filesystem nor a remote path, they still save all of the data
            #   * its easier to implement and avoid race condition on the shared data.
            ckpt_path = os.path.join(ckpt_path, f"diloco_{self.world_info.diloco_rank}")

        catch_warning = self._logger.getEffectiveLevel() <= logging.INFO

        with warnings.catch_warnings():
            # pytorch has an annoying warning when saving the optimizer state https://github.com/pytorch/pytorch/issues/136907
            # we can ignore it if we are not logging in DEBUG mode
            if catch_warning:
                warnings.simplefilter("ignore")

            dcp.save(self.states, checkpoint_id=ckpt_path)

            ## the next part is a fix so that each rank save a different dataloader rank. It not efficient because it reads the state two times from disk
            with open(os.path.join(ckpt_path, f"__{self.world_info.local_rank}_0.pt"), "wb") as f:
                state = {"data_loader": self.dataloader.state_dict()}
                if self.diloco_offloaded_optimizer:
                    state["optimizer"] = OuterOptimizerWrapper(self.diloco_offloaded_optimizer).state_dict()

                torch.save(state, f)

        gc.collect()

    def _async_save_remote(self, ckpt_path: str, remote_ckpt_path: str, second_remote_path: str | None = None):
        """asyncronously rsync a ckpt folder to a remote location. Using fsspec to handle remote cloud storage without to install
        specific libraries (e.g. s3fs).

        If second_remote_path is provided, it will rsync the remote_ckpt_path to the second_remote_path after the first. sync
        Usefull for sh to disk to remote save operation.
        """

        def _rsync(src_path: str, dest_path: str):
            time_start = time.perf_counter()
            self._logger.info(f"start pushing {src_path} to {dest_path} asynchronously")
            rsync_fsspec(src_path, destination=dest_path)
            self._logger.info(f"finish pushing {src_path} to {dest_path} in {time.perf_counter() - time_start} seconds")

        def rsync():
            _rsync(ckpt_path, remote_ckpt_path)
            if second_remote_path:
                _rsync(remote_ckpt_path, second_remote_path)

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
        if self.live_server is not None:
            shutil.rmtree(self.shm_path, ignore_errors=True)
            self.live_server.stop()

        self.wait_async_save_process()

    def load(self, resume_ckpt_path: str, diloco_rank: int | None = None, skip_dataloader: bool = False) -> None:
        """
        loading should be done after fsdp wrap and optimizer init.
        Each rank will load the right shard of the model and optimizer.
        All rank will load the global states (scheduler, step, total_tokens, dataloader).

        `resume_ckpt_path` should point to a specific step and not to the base ckpt folder. Example: `ckpt_path/step_100`

        Loading is done inplace.

        direct_diloco_folder = False. mean that `diloco_rank` is added to the resume_ckpt_path.
        """
        time_start = time.perf_counter()

        world_info = get_world_info()
        if self.diloco_offloaded_param_list is not None:
            rank = diloco_rank if diloco_rank is not None else world_info.diloco_rank
            resume_ckpt_path = os.path.join(resume_ckpt_path, f"diloco_{rank}")

        dcp.load(self.states, checkpoint_id=resume_ckpt_path)

        self._logger.debug("sync inner model")
        # todo(refactor): here we should rather let the diloco class handle this logic
        for param_offloaded, param in zip(self.diloco_offloaded_param_list, self.model.parameters()):
            param_offloaded.data.to_local().copy_(param.data.to_local())

        if not skip_dataloader:
            ## the next part is a fix so that each rank save a different dataloader rank. It not efficient because it reads the state two times from disk
            with open(os.path.join(resume_ckpt_path, f"__{world_info.local_rank}_0.pt"), "rb") as f:
                rank_state_dict = torch.load(f)

            self.dataloader.load_state_dict(rank_state_dict["data_loader"])

        if self.diloco_offloaded_optimizer:
            opt_wrapper = OuterOptimizerWrapper(self.diloco_offloaded_optimizer)
            opt_wrapper.load_state_dict(rank_state_dict["optimizer"])

        self._init_state()

        self._logger.info(f"Loaded checkpoint from {resume_ckpt_path} in {time.perf_counter() - time_start} seconds")

    def download_and_load_ckpt_from_peers(self, address: str):
        time_start = time.perf_counter()
        ckpt_path = f"/dev/shm/zeroband_reco/node_{self.world_info.global_rank}"
        path = os.path.join(ckpt_path, f"diloco_{self.world_info.diloco_rank}")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        dest_rank = 0

        if self.world_info.local_rank == 0:
            # only local rank download the ckpt
            self._logger.info(f"Started downloading ckpt from http://{address}/diloco_{dest_rank}")
            wget(
                source=f"http://{address}/diloco_{dest_rank}",
                destination=path,
            )
            wget(
                source=f"http://{address}/diloco_{dest_rank}/.metadata",
                destination=path,
            )
        dist.barrier()
        self.load(resume_ckpt_path=ckpt_path, skip_dataloader=True)
        self._logger.info(
            f"Downloaded checkpoint from http://{address}/diloco_{dest_rank} in {time.perf_counter() - time_start} seconds"
        )
        # we don't want the dataloader states to be loaded as they are not the same on each rank


class CkptLiveServer:
    def __init__(self, port: int, ckpt_path: str):
        self.port = port
        self.ckpt_path = ckpt_path
        self._logger = get_logger()

        self._process = multiprocessing.Process(target=self._start_http_server, daemon=True)
        self._process.start()

    def _start_http_server(self):
        import http.server
        import socketserver

        os.makedirs(self.ckpt_path, exist_ok=True)
        os.chdir(self.ckpt_path)
        with socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
            self._logger.debug(f"Start serving live ckpt on {self.port}")
            httpd.serve_forever()

    def stop(self):
        self._process.terminate()

    def __del__(self):
        self.stop()
