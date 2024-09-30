# ZeroBand
ZeroBand is a production ready codebase for decentralized training of LLM


## Development

install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

Optionnaly create a venv

```
uv venv
source .venv/bin/activate
```

Install deps
```
uv sync --extra all
```


copy paste to full command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv venv
source .venv/bin/activate
uv sync --extra all
```


run your code using 

```bash
uv run ...
```

## Quick check

To check that everything is working you can do

```bash
ZERO_BAND_LOG_LEVEL=DEBUG torchrun --nproc_per_node=2 src/zeroband/train.py @configs/debug/normal.toml
```

## Run diloco

To run diloco locally you can use the helper script `scripts/simulatsimulate_multi_nodee_mutl.sh` 

:note: you need 4 gpus to run the following command

```bash
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 2 src/zeroband/train.py @configs/debug/diloco.toml
```

if you have only two gpus

```bash
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 1 src/zeroband/train.py @configs/debug/diloco.toml
```

One gpu is not supported at the moment because of a fsdp bug in our implementation.

## run test

You need a machine with a least two gpus to run the full test suite.

Some test must be run from the root directory.
```bash
uv run pytest
```

## Potential foot gun to avoid:

if you have a datasets error at the beginning of training try to use the following env var
```
HF_HUB_ETAG_TIMEOUT=500
```

## Environment variables
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `GLOBAL_UNIQUE_ID`    | Unique identifier worker in global store.        | `None`  |
| `GLOBAL_ADDR`         | IP Address of the global store                   | `None`  |
| `GLOBAL_PORT`         | Port number of the global store.                 | `None` |
| `GLOBAL_WORLD_SIZE`   | The size of the global process group.            | `1` |
| `GLOBAL_RANK`         | Rank of the process in the global process group. | `0` |
| `ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS` | Number of seconds before the global store operations timeout | `300` |
