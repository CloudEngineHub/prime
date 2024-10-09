import torch
from zeroband.data import SequencePackingDataSet
from torch.utils.data import DataLoader
from zeroband.data import load_all_datasets, DataConfig, logger as data_logger
from collections import Counter
from itertools import chain
import pytest
import logging


@pytest.mark.parametrize(
    "ratio, lower, upper",
    [
        ("3:2", 1.2821, 1.7549),
        ("0.5:1", 0.4247, 0.5886),
    ],
)
def test_load_all_datasets_vanilla(ratio: str, lower: float, upper: float):
    config = DataConfig(
        dataset_name_or_paths="Jackmin108/abc-testing:A,Jackmin108/abc-testing:C",
        dataset_ratio=ratio,
        streaming=True,
        fake=False,
    )

    ds = load_all_datasets(config, "train")
    print(ds)

    dl = DataLoader(ds, batch_size=256)
    batches = [i["text"] for i, _ in zip(dl, range(10))]
    assert len(batches) == 10

    # Check that the ratio is correct
    letter_count = Counter(i[0] for i in chain(*batches))
    print(letter_count, letter_count["A"] / letter_count["C"])
    assert letter_count["A"] / letter_count["C"] < upper
    assert letter_count["A"] / letter_count["C"] > lower


@pytest.mark.parametrize(
    "ratio, lower, upper, data_rank, data_world_size",
    [
        ("3:2", 1.2821, 1.7549, 1, 4),
        ("0.5:1", 0.4247, 0.5886, 0, 3),
    ],
)
def test_load_all_datasets_data_rank(ratio: str, lower: float, upper: float, data_rank: int, data_world_size: int):
    data_logger.setLevel(logging.DEBUG)
    config = DataConfig(
        dataset_name_or_paths="Jackmin108/abc-testing:A,Jackmin108/abc-testing:C",
        dataset_ratio=ratio,
        streaming=True,
        fake=False,
        data_world_size=data_world_size,
        data_rank=data_rank,
    )

    ds = load_all_datasets(config, "train")
    print(ds)

    dl = DataLoader(ds, batch_size=256)
    batches = [i["text"] for i, _ in zip(dl, range(10))]
    assert len(batches) == 10

    # Check that the ratio is correct
    letter_count = Counter(i[0] for i in chain(*batches))
    print(letter_count, letter_count["A"] / letter_count["C"])
    assert letter_count["A"] / letter_count["C"] < upper
    assert letter_count["A"] / letter_count["C"] > lower

    c_num_set = {int(i[1:]) for i in chain(*batches) if i[0] == "C"}
    a_num_set = {int(i[1:]) for i in chain(*batches) if i[0] == "A"}

    # Check that the data is correctly sharded
    first_a_shard = set(range(data_rank * (2**12), (data_rank + 1) * (2**12)))
    first_10_c_shard = set()
    for i in range(data_rank, data_world_size * 10, data_world_size):
        first_10_c_shard = first_10_c_shard.union(set(range(i * (2**8), (i + 1) * (2**8))))
    assert all(i in first_a_shard for i in a_num_set)
    assert all(i in first_10_c_shard for i in c_num_set)


def test_squence_packing():
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [[6, 1, 2, 3, 4], [6, 3, 3, 4, 1, 7], [3, 2], [1, 2], [1, 4, 5, 3, 4, 1, 7, 8]]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return {'input_ids': self.data[index]}

    MAX_SEQ_LEN = 8
    dataset = SequencePackingDataSet(FakeDataset(), max_seq_length=MAX_SEQ_LEN, eos_token=0)

    input_ids = []
    labels = []
    for data in dataset:
        assert data["input_ids"].shape[0] == MAX_SEQ_LEN
        assert data["labels"].shape[0] == MAX_SEQ_LEN
        assert sum(data["seqlens"]) == MAX_SEQ_LEN

        input_ids.append(data["input_ids"].tolist())
        labels.append(data["labels"].tolist())

    assert input_ids == [[6, 1, 2, 3, 4, 6, 3, 3], [3, 2, 1, 2, 1, 4, 5, 3]]
    assert labels == [[1, 2, 3, 4, 0, 3, 3, 4], [2, 0, 2, 0, 4, 5, 3, 4]]
