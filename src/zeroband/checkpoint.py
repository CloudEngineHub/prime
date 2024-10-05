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
from zeroband.comms import LIVE_RECO_PORT
from zeroband.utils.logging import get_logger
import warnings
import logging
from zeroband.utils.wget import wget

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
        ckpt_path: str | None,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        dataloader: StatefulDataLoader,
        training_progress: TrainingProgress,
        diloco_offloaded_param_list: list[nn.Parameter] | None,
        diloco_offloaded_optimizer: Optimizer | None,
        live_ckpt_server: bool = False,
    ):
        self.ckpt_path = ckpt_path

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
            self.live_server = CkptLiveServer(port=LIVE_RECO_PORT + self.world_info.global_rank, ckpt_path=SHM_PATH)

    def _init_state(self):
        # states can only be stateful object, hence we need to wrap Model and Optimizer
        self.states: dict[str, Stateful] = {
            "model": ModelWrapper(self.model),
            "optimizer": OptimizerWrapper(self.model, self.optimizer),
            "scheduler": self.scheduler,
            # "dataloader": self.dataloader, # ignoring dataloader for now as each rank has its own dataloader
            "training_progress": self.training_progress,
        }

        if self.diloco_offloaded_optimizer is not None:
            # even if the diloco_offloaded target the cpu list model, we still use the gpu model to load and save state.
            # main reason is that we actually don't a cpu model but just a list of cpu parameters.
            self.states["diloco_optimizer"] = self.diloco_offloaded_optimizer

    def save_shm(self) -> None:
        """
        Save the latest checkpoint in shared memory.
        """
        self.save(overide_ckpt_path=SHM_PATH, remote_ckpt_path=None)

    def save(self, remote_ckpt_path: str | None, overide_ckpt_path: str | None = None) -> None:
        """
        Each rank will save the right shard of the model and optimizer.

        Saving is done inplace.

        Save in the subfolder `step_<step>` and create a symlink `latest`.
        """

        time_start = time.perf_counter()
        world_info = get_world_info()

        if not overide_ckpt_path:
            og_ckpt_path = self.ckpt_path
            ckpt_path = os.path.join(og_ckpt_path, f"step_{self.training_progress.step}")
        else:
            og_ckpt_path = overide_ckpt_path
            ckpt_path = os.path.join(overide_ckpt_path, "latest")

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

            if overide_ckpt_path is None:
                ## create a symlink from step_{now} to latest
                latest_link = os.path.join(og_ckpt_path, "latest")
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                os.symlink(f"step_{self.training_progress.step}", latest_link)

        self._logger.info(f"Saved checkpoint to {ckpt_path} in {time.perf_counter() - time_start} seconds")

        gc.collect()

        if remote_ckpt_path is not None:
            self._async_save_remote(ckpt_path, remote_ckpt_path)

    def _async_save_remote(self, remote_ckpt_path: str):
        """asyncronously rsync a ckpt folder to a remote location. Using fsspec to handle remote cloud storage without to install
        specific libraries (e.g. s3fs)
        """

        def rsync():
            time_start = time.perf_counter()
            self._logger.info(f"start pushing {self.ckpt_path} to {remote_ckpt_path} asynchronously")
            rsync_fsspec(self.ckpt_path, destination=remote_ckpt_path)
            self._logger.info(
                f"finish pushing {self.ckpt_path} to {remote_ckpt_path} in {time.perf_counter() - time_start} seconds"
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
        os.remove(SHM_PATH)
        self.wait_async_save_process()
        if self.live_server is not None:
            self.live_server.stop()

    def load(self, resume_ckpt_path: str, diloco_rank: int | None = None) -> None:
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
        # since we don't load the param list from the state dict as its the same as the model one we just copy
        if self.diloco_offloaded_param_list is not None:
            for param_offloaded, param_model in zip(self.diloco_offloaded_param_list, self.model.parameters()):
                param_offloaded.data.copy_(param_model.data)

        ## the next part is a fix so that each rank save a different dataloader rank. It not efficient because it reads the state two times from disk
        with open(os.path.join(resume_ckpt_path, f"__{world_info.local_rank}_0.pt"), "rb") as f:
            rank_state_dict = torch.load(f)

        self.dataloader.load_state_dict(rank_state_dict["data_loader"])

        self._init_state()

        self._logger.info(f"Loaded checkpoint from {resume_ckpt_path} in {time.perf_counter() - time_start} seconds")

    def download_and_load_ckpt_from_peers(self, adress: str):
        path = f"/tmp/zeroband/node_{self.world_info.global_rank}"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        dest_rank = self.world_info.global_rank - 1

        if self.world_info.local_rank == 0:
            # only local rank download the ckpt
            wget(
                source=f"http://{adress}/latest/diloco_{dest_rank}",
                destination=path,
            )
            wget(
                source=f"http://{adress}/latest/diloco_{dest_rank}/.metadata",
                destination=path,
            )
        dist.barrier()
        self.load(resume_ckpt_path=path, diloco_rank=dest_rank)


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
