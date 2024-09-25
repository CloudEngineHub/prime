# ZeroBand
ZeroBand is a production ready codebase for decentralized training of LLM


## developlment

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

## quick check

To check that everything is working you can do

```bash
ZERO_BAND_LOG_LEVEL=DEBUG torchrun  --nproc_per_node=2 src/zeroband/train.py @configs/debug/normal.toml
```

## run diloco

To run diloco locally you can use the helper script `scripts/simulatsimulate_multi_nodee_mutl.sh`

```bash
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node.sh 2 2 src/zeroband/train.py @configs/debug/diloco.toml
```

## run test

You need a machine with a least two gpus to run the full test suite.

Some test must be run from the root directory.

```bash
uv run pytest
```

