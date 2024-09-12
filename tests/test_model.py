import pytest
import torch
from zeroband.models.llama import Transformer, llama2_configs


VOCAB_SIZE = 1024

@pytest.fixture
def llama_config():
    config =  llama2_configs["debugmodel"]
    config.vocab_size = VOCAB_SIZE
    return config

def test_llama(llama_config):
    seq_len = 512
    bs = 8
    model = Transformer(llama_config)
    input_ = torch.randint(0, llama_config.vocab_size, (bs, seq_len))
    output = model(input_)
    assert output.shape == (bs, seq_len, llama_config.vocab_size)

