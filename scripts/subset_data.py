#!/usr/bin/env python
# coding: utf-8
# Usage:
# python scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size 12 --data_rank 1

import argparse
from typing import Dict, List, Optional
import functools
from datasets import load_dataset_builder, BuilderConfig
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@functools.lru_cache(maxsize=None)
def _get_ds_config_dict(path: str, name: Optional[str] = None) -> Dict[str, BuilderConfig]:
    ds_builder = load_dataset_builder(path=path, name=name)
    return ds_builder.builder_configs


def _get_datafiles(path: str, name: Optional[str] = None, split: str = "train") -> List[str]:
    builder_config = _get_ds_config_dict(path=path, name=name)
    if name is None:
        if "default" not in builder_config:
            name = next(iter(builder_config.keys()))
        else:
            name = "default"
    return builder_config[name].data_files[split]


def main(args):
    g_data_files = _get_datafiles(args.dataset_name)
    logger.debug(f"Length of data_files: {len(g_data_files)}")
    if len(args.filter) > 0:
        args.filter = args.filter.split(",")
        data_files = []
        for _filter in args.filter:
            data_files.extend([f for f in g_data_files if _filter in f])
    else:
        data_files = g_data_files

    logger.debug(f"Length of data_files: {len(data_files)}")
    data_files = set(data_files[args.data_rank :: args.data_world_size][: args.max_shards])
    logger.debug(f"Data files: {data_files}")
    logger.debug(f"Length of data_files processing: {len(data_files)}")

    script_path = Path("./rm-unused.sh")
    with script_path.open("w") as f:
        f.write("#!/bin/bash\n")
        for data_file in g_data_files:
            if data_file not in data_files:
                f.write(f"git rm {'/'.join(data_file.split('@')[-1].split('/')[1:])}\n")
        f.write("git commit -m 'remove unused data files'\n")
        f.write("git lfs pull -I '*.parquet'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process data from the Stack V2 dataset")
    parser.add_argument("--dataset_name", type=str, default="dwkdwh", help="dataset name")
    parser.add_argument("--dry_run", action="store_true", help="do not download data")
    parser.add_argument("--filter", type=str, default="", help="search shards by the filter")
    parser.add_argument("--data_rank", type=int, default=0, help="start index")
    parser.add_argument("--data_world_size", type=int, default=4, help="world size")
    parser.add_argument("--max_shards", type=int, default=1000)
    args = parser.parse_args()
    main(args)
