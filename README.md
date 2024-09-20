# ZeroBand
ZeroBand is a production ready codebase for decentralized training of LLM


## developlment

install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra all
```

run your code using 

```bash
uv run ...
```

## quick check

To check that everything is working you can do

```bash
ZERO_BAND_LOG_LEVEL=DEBUG torchrun  --nproc_per_node=2 src/zeroband/train.py @configs/debug.toml
```

## run test

You need a machine with a least two gpus to run the full test suite.

Some test must be run from the root directory.

```bash
uv run pytest
```

