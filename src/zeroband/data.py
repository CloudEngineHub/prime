from functools import partial
from typing import Any, Generator, Optional, List, Dict, Union
from pydantic_config import BaseConfig
from zeroband.utils.logging import get_logger

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from datasets import load_dataset, interleave_datasets, load_dataset_builder, BuilderConfig
from datasets.distributed import split_dataset_by_node
import functools

TEST_VOCAB_SIZE = 1024

# TODO sami: make sure the init of the model is the same on all rank

logger = get_logger(__name__)


class DataConfig(BaseConfig):
    dataset_name_or_paths: str = "allenai/c4:en"
    val_dataset_name_or_paths: Optional[str] = None
    seq_length: int = 1024
    fake: bool = False
    num_workers: int = 4
    streaming: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    dataset_ratio: Optional[str] = None
    data_rank: Optional[int] = None
    data_world_size: Optional[int] = None


class FakeTokenizedDataset(IterableDataset):
    """This is a dummy dataset that generates random sequences of length seq_len and vocab_size"""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"

    def __iter__(self) -> Generator[dict[str, Any], Any, None]:
        while True:
            input_ids = torch.randint(3, self.vocab_size, (self.seq_len,)).tolist()
            yield {"input_ids": input_ids}


def collate_causal_mask(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100) -> callable:
    """collate function for causal mask. Fill with padding tokens if sequence is shorter than max_seq_length"""
    return partial(_collate_fn_causal_mask, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)


def _collate_fn_causal_mask(
    samples: list[dict[str, torch.LongTensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
) -> dict[str, torch.LongTensor]:
    """collate function for causal mask. Fill with padding tokens if sequence is shorter than max_seq_length.
    input_ids and labels are both of size max_seq_length.
    """

    assert samples[0].keys() == {"input_ids"}

    batched = {"input_ids": [], "labels": []}

    if max_seq_length > 0:
        max_seq_length += 1  # this makes sure that the effective seqlen is correct

    for sample in samples:
        input_ids = torch.Tensor(sample["input_ids"]).long()

        if len(input_ids) < max_seq_length:
            input_ids = torch.cat([input_ids, torch.full((max_seq_length - len(input_ids),), pad_id)])
        elif len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        batched["input_ids"].append(input_ids[:-1])
        batched["labels"].append(input_ids[1:])

    return {"input_ids": torch.stack(batched["input_ids"], dim=0), "labels": torch.stack(batched["labels"], dim=0)}


def get_dataloader(
    tokenizer, world_size: int, rank: int, batch_size: int, data_config: DataConfig, pad_token_id: int
) -> DataLoader:
    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, TEST_VOCAB_SIZE)
    else:
        ds = load_all_datasets(data_config=data_config, split="train")

        def tokenize_function(data):
            outputs = tokenizer(data["text"], truncation=True, max_length=data_config.seq_length)
            return outputs

        tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "attention_mask"])
        train_dataset = split_dataset_by_node(tokenized_datasets, world_size=world_size, rank=rank)

    data_collator = collate_causal_mask(max_seq_length=data_config.seq_length, pad_id=pad_token_id, ignore_index=-100)

    return StatefulDataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=data_config.num_workers,
    )


@functools.lru_cache(maxsize=None)
def _get_ds_config_dict(path: str, name: Optional[str] = None) -> Dict[str, BuilderConfig]:
    ds_builder = load_dataset_builder(path=path, name=name)
    return ds_builder.builder_configs


def _get_datafiles(path: str, name: Optional[str] = None, split: str = "train") -> List[str]:
    builder_config = _get_ds_config_dict(path=path, name=name)
    if name is None:
        if "default" not in builder_config:
            logger.warning(f"Default config not found for {path}. Using first config.")
            name = next(iter(builder_config.keys()))
        else:
            name = "default"
    return builder_config[name].data_files[split]


def _nice_print(kwargs: Dict[str, Union[str, List[str]]]) -> str:
    def _foo(a):
        if isinstance(a, list):
            return str(a[:5]) + "..." + str(a[-5:]) if len(a) > 10 else str(a)
        return str(a)

    return str({k: _foo(v) for k, v in kwargs.items()})


def _load_datasets(
    dataset_names: str,
    split: str,
    data_rank: Optional[int] = None,
    data_world_size: Optional[int] = None,
    streaming: bool = True,
    probabilities: Optional[List[float]] = None,
) -> Dataset:
    logger.debug(dataset_names)
    ds_args = []
    for _ds in dataset_names.split(","):
        _ds_name, _, _ds_config = _ds.partition(":")
        _ds_args = {"path": _ds_name}
        if _ds_config:
            _ds_args["name"] = _ds_config
        if data_rank is not None and data_world_size is not None:
            _data_files = _get_datafiles(_ds_name, _ds_config, split)
            _ds_args["data_files"] = _data_files[data_rank::data_world_size]
        ds_args.append(_ds_args)

    logger.debug(f"Datasets ({split}):\n" + "\n".join(map(_nice_print, ds_args)))
    logger.debug(f"Probabilities: {probabilities}")
    logger.debug(f"Loading datasets{' in streaming mode' if streaming else ''}")
    datasets = []
    for ds_arg in ds_args:
        logger.debug(f"Loading dataset: {ds_arg}")
        _ds = load_dataset(**ds_arg, split=split, streaming=streaming)
        _ds = _ds.remove_columns([i for i in _ds.column_names if i not in ["text"]])
        datasets.append(_ds)
        logger.debug(f"Loaded dataset: {ds_arg}")

    ds = interleave_datasets(
        datasets=datasets,
        probabilities=probabilities,
    )
    logger.info(f"Loaded datasets ({split})")
    return ds


def _get_probabilities(data_config: DataConfig) -> Optional[List[float]]:
    if data_config.dataset_ratio is None:
        return None
    if len(data_config.dataset_name_or_paths.split(",")) != len(data_config.dataset_ratio.split(":")):
        raise ValueError("Number of datasets and dataset ratios must be the same")
    nums = [float(i) for i in data_config.dataset_ratio.split(":")]
    denom = sum(nums)
    return [i / denom for i in nums]


def load_all_datasets(data_config: DataConfig, split: str, max_samples: Optional[int] = None) -> IterableDataset:
    """Load all datasets and interleave them"""
    if max_samples is not None and not data_config.streaming:
        split = f"{split}[:{max_samples}]"
    ds = _load_datasets(
        dataset_names=data_config.dataset_name_or_paths,
        split=split,
        data_rank=data_config.data_rank,
        data_world_size=data_config.data_world_size,
        streaming=data_config.streaming,
        probabilities=_get_probabilities(data_config),
    )
    if max_samples is not None and data_config.streaming:
        if data_config.max_train_samples is not None:
            ds = ds.take(data_config.max_train_samples)
    logger.info(f"Train dataset:\n{ds}")

    return ds
