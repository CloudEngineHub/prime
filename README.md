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
## On off ramping routines
- For the first initialisation, all the GLOBAL env vars matter and will be used by the nodes to initialize.
- When nodes join, only the `GLOBAL_ADDR` and `GLOBAL_PORT` matter. You still have to set `GLOBAL_RANK` and `GLOBAL_WORLD_SIZE` but they will be updated when the global pg initializes.
- When a node wishes to offboard, it must call `edm._queue_leave()` and then `edm.maybe_reinit_global_pg()`. The mechanism is that it has to tell the master it is leaving and then join the the next `edm.maybe_reinit_global_pg()` in order to not deadlock the barrier for master's `_resolve_world()`. Jackmin is trying to change this behavior such that the leaving node can leave without having to `edm.maybe_reinit_global_pg()` or without `edm._queue_leave()` but they are required for now.

## Environment variables
### Global Store Initialization
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `GLOBAL_UNIQUE_ID`    | Unique identifier worker in global store.        | `None`  |
| `GLOBAL_ADDR`         | IP Address of the global store                   | `None`  |
| `GLOBAL_PORT`         | Port number of the global store.                 | `None` |
| `GLOBAL_WORLD_SIZE`   | The size of the global process group.            | `1` |
| `GLOBAL_RANK`         | Rank of the process in the global process group. | `0` |

### Elastic Device Mesh Configuration
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `ZERO_BAND_LOG_LEVEL` | Enable debug mode for loge | `False` |
| `ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS` | Number of seconds before the global store operations timeout | `300` |
| `ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS` | Number of seconds between polls to the store when waiting for values | `0.1` |
| `ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS` | Interval in seconds between heartbeats | `2` |
| `ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS` | Time in seconds after which a node is considered dead if no heartbeat is received | `10` |
