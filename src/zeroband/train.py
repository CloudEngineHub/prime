import os
from contextlib import nullcontext
from typing import Literal

import torch
from pydantic_config import parse_argv, BaseConfig
from einops import rearrange
from torch.nn import functional as F

from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
import torch.distributed as dist
from zeroband import utils
from zeroband.diloco import Diloco, DilocoConfig
from zeroband.comms import ElasticDeviceMesh

from zeroband.utils import PerfCounter, get_module_signature, get_sharding_strategy
from zeroband.utils.monitor import WandbMonitor, DummyMonitor
from zeroband.data import TEST_VOCAB_SIZE, get_dataloader
from zeroband.models.llama import get_model
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
from zeroband.checkpoint import CkptManager, TrainingProgress


class DataConfig(BaseConfig):
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4


class OptimConfig(BaseConfig):
    lr: float = 4e-4
    weight_decay: float = 0.1
    adam_betas1: float = 0.9
    adam_betas2: float = 0.95

    warmup_steps: int = 1000
    total_steps: int = 88_000
    batch_size: int = 512


class TrainConfig(BaseConfig):
    micro_bs: int
    torch_compile: bool = True
    sharding_strategy: str = "SHARD_GRAD_OP"

    log_model_hash: bool = False


class CkptConfig(BaseConfig):
    path: str
    interval: int

    remote_path: str | None = None  # could be a s3 path


class Config(BaseConfig):
    # main config
    name_model: Literal["debugmodel", "150M", "271M", "1B", "7B", "13B", "26B", "70B"] = "150M"
    type_model: Literal["llama2", "llama3"] = "llama2"

    project: str = "zeroband"
    metric_logger_type: Literal["wandb", "dummy"] = "wandb"

    # sub config
    diloco: DilocoConfig | None = None
    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    train: TrainConfig

    ckpt: CkptConfig | None = None
    resume: str | None = None


def train(config: Config):
    sharding_strategy = get_sharding_strategy(config.train.sharding_strategy)

    # batch_size is the total batch size for all GPUs
    assert config.optim.batch_size % world_info.local_world_size == 0
    batch_size = config.optim.batch_size // world_info.local_world_size

    assert batch_size % config.train.micro_bs == 0
    gradient_accumulation_steps = batch_size // config.train.micro_bs

    if config.ckpt is not None and config.ckpt.interval is not None and config.diloco is not None:
        assert (
            config.ckpt.interval % config.diloco.inner_steps == 0
        ), "ckpt interval must be a multiple of diloco inner steps as we only save at the end of an outer step"

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"  # todo(sami): remove padding tokens once we have context stuffing

    logger.debug("tokenizer loaded")

    train_dataloader = get_dataloader(
        tokenizer=tokenizer,
        world_size=world_info.world_size * world_info.global_world_size,
        rank=world_info.rank + world_info.global_rank * world_info.global_world_size,
        seq_length=config.data.seq_length,
        batch_size=config.train.micro_bs,
        num_workers=config.data.num_workers,
        fake_data=config.data.fake,
    )

    model, model_config = get_model(
        config.name_model,
        config.type_model,
        vocab_size=tokenizer.vocab_size
        if config.name_model != "debugmodel" or not config.data.fake
        else TEST_VOCAB_SIZE,
    )

    if config.train.log_model_hash:
        # Compute SHA256 hash
        logger.info(f"Model hash: {get_module_signature(model)}")

    model = model.to(world_info.local_rank)
    logger.debug("model loaded")

    gpu_peak_flops = utils.get_peak_flops(torch.cuda.get_device_name(torch.device("cuda")))
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    num_params = utils.get_num_params(model, exclude_embedding=True)
    logger.info(f"Number of parameters: {num_params}")
    num_flop_per_token = utils.get_num_flop_per_token(
        num_params,
        model_config,
        config.data.seq_length,
    )

    elastic_device_mesh = ElasticDeviceMesh()

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
        use_orig_params=True,
        process_group=elastic_device_mesh.local_pg if config.diloco is not None else None,
    )
    logger.debug("model fsdped")

    # Setup optimizers
    inner_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        betas=(config.optim.adam_betas1, config.optim.adam_betas2),
    )

    if config.diloco is not None:
        diloco = Diloco(config.diloco, model, sharding_strategy, elastic_device_mesh.global_pg)

    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=config.optim.warmup_steps,
        num_training_steps=config.optim.total_steps,
    )

    training_progress = TrainingProgress(total_tokens=0, outer_step=0, step=0)

    ckpt_manager = CkptManager(
        model=model,
        optimizer=inner_optimizer,
        scheduler=scheduler,
        dataloader=train_dataloader,
        training_progress=training_progress,
        diloco_offloaded_optimizer=diloco.outer_optimizer if config.diloco is not None else None,
        diloco_offloaded_param_list=diloco.param_list_cpu if config.diloco is not None else None,
    )

    if config.train.torch_compile:
        # we need to compile AFTER creating the CKPT manager, DON'T ASK ME WHY
        model = torch.compile(model)
        logger.debug("model compiled")

    if config.resume is not None:
        # all is inplace
        ckpt_manager.load(resume_ckpt_path=config.resume)

    model.train()

    if world_info.rank == 0:
        logger_cls = WandbMonitor if config.metric_logger_type == "wandb" else DummyMonitor
        metric_logger = logger_cls(project=config.project, config=config.model_dump(), resume=False)

    train_dataloader_iterator = iter(train_dataloader)

    num_inner_steps = config.diloco.inner_steps if config.diloco is not None else 1
    perf_counter = PerfCounter(window_size=10)

    logger.info("starting training")
    while True:
        if num_inner_steps > 1:
            # if we don't use diloco we don't print the outer step logs
            logger.info(f"outer_step step: {training_progress.outer_step}")

        for _inner_step in range(num_inner_steps):
            loss_batch = 0

            for grad_acc_step in range(gradient_accumulation_steps):
                is_accumulating = grad_acc_step < gradient_accumulation_steps - 1
                batch = next(train_dataloader_iterator)
                input_ids = batch["input_ids"].to("cuda")
                labels = batch["labels"].to("cuda")

                with model.no_sync() if is_accumulating else nullcontext():
                    logits = model(tokens=input_ids).contiguous()
                    flatten_logits = rearrange(logits, "b seq vocab -> (b seq) vocab")
                    flatten_labels = rearrange(labels, "b seq -> (b seq)")

                    loss = (
                        F.cross_entropy(flatten_logits, flatten_labels, ignore_index=tokenizer.pad_token_id)
                        / gradient_accumulation_steps
                    )
                    loss.backward()
                    loss_batch += loss.detach()

            model.clip_grad_norm_(1.0)  # gradient clipping
            inner_optimizer.step()
            scheduler.step()
            inner_optimizer.zero_grad()

            # logging
            training_progress.step += 1
            inner_lr = [group["lr"] for group in inner_optimizer.param_groups][0]

            dist.all_reduce(tensor=loss_batch, op=dist.ReduceOp.AVG, group=elastic_device_mesh.local_pg)
            # syncing loss across all data parallel rank within a nodes

            new_tokens = config.data.seq_length * config.optim.batch_size
            perf_counter.count_tokens(new_tokens)

            if config.diloco is not None:
                training_progress.total_tokens += new_tokens
            else:
                # we count the total tokens with respect to all diloco workers
                # might need to tweak this as some worker might fail to join the all reduce later
                training_progress.total_tokens += new_tokens * elastic_device_mesh.global_pg.size()

            metrics = {
                "Loss": loss_batch.item(),
                "step": training_progress.step,
                "inner_lr": inner_lr,
                "Perplexity": torch.exp(loss_batch).item(),
                "total_tokens": training_progress.total_tokens,
            }
            log = f"step: {training_progress.step}, loss: {loss_batch.item():.4f}"

            tokens_per_second = perf_counter.get_tokens_per_second()

            if tokens_per_second is not None:
                metrics["tokens_per_second"] = tokens_per_second
                metrics["mfu"] = (
                    100 * num_flop_per_token * tokens_per_second / gpu_peak_flops / world_info.local_world_size
                )
                log += f", tokens_per_second: {tokens_per_second:.2f}, mfu: {metrics['mfu']:.2f}"

            if config.diloco is not None:
                metrics["num_peers"] = elastic_device_mesh.global_pg.size()
                log += f", diloco_peers: {metrics['num_peers']}"

            if world_info.rank == 0:
                metric_logger.log(metrics)

            logger.info(log)

        if config.diloco is not None:
            if config.train.log_model_hash:
                with FSDP.summon_full_params(model):
                    logger.debug("Pre diloco model: %s", get_module_signature(model))
            diloco.step(model)
            if config.train.log_model_hash:
                with FSDP.summon_full_params(model):
                    logger.debug("Post diloco model: %s", get_module_signature(model))

        training_progress.outer_step += 1

        if (
            config.ckpt is not None
            and training_progress.step > 0
            and training_progress.step % config.ckpt.interval == 0
        ):
            # we only allow to checkpoint after a outer step. For non diloco training outer step = 1 anyway
            ckpt_manager.save(config.ckpt.path, config.ckpt.remote_path)

        if training_progress.step >= config.optim.total_steps:
            # we only allow to break outisde of the inner loop.
            # This avoid ending the training in the middle of a the inner loop
            # Since ckpt strategy and all reduce is done at the outer loop level.
            break

    if world_info.rank == 0:
        metric_logger.finish()

    ckpt_manager.wait_async_save_process()
    logger.info("Training finished, exiting ...")


if __name__ == "__main__":
    # Allow eager fallback during production so that that the training runs dont die
    # However, in development, we want to know that we broke torch compile
    torch._dynamo.config.suppress_errors = "ZERO_BAND_DEV" not in os.environ
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)  # this ensure same weight init across diloco workers

    world_info = get_world_info()
    logger = get_logger()

    torch.cuda.set_device(world_info.local_rank)

    config = Config(**parse_argv())
    logger.debug(f"config: {config.model_dump()}")

    train(config)
