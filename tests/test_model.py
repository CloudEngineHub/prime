import pytest
import torch
from zeroband.models.llama import Transformer, llama2_configs
from zeroband.models.llama.model import Attention, ModelArgs


VOCAB_SIZE = 1024

ERROR_ATOL = {
    torch.float: 3e-4,
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL = {
    torch.float: 2e-5,
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}


@pytest.fixture
def llama_config() -> ModelArgs:
    config = llama2_configs["debugmodel"]
    config.vocab_size = VOCAB_SIZE
    return config


@pytest.mark.parametrize("attn_fn", ["flash", "sdpa"])
def test_llama(llama_config: ModelArgs, attn_fn):
    seq_len = 512
    bs = 8
    llama_config.attn_fn = attn_fn
    model = Transformer(llama_config).to("cuda")
    input_ = torch.randint(0, llama_config.vocab_size, (bs, seq_len)).to("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_)

    assert output.shape == (bs, seq_len, llama_config.vocab_size)


def get_freqs_cis(llama_config: ModelArgs):
    model = Transformer(llama_config).to("cuda")
    return model.freqs_cis


def test_attn(llama_config: ModelArgs):
    seq_len = 512
    bs = 8

    freqs_cis = get_freqs_cis(llama_config)
    input_ = torch.rand(bs, seq_len, llama_config.dim).to("cuda")

    attn = Attention(llama_config).to("cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_sdpa = attn(input_, freqs_cis)

    attn.attn_fn = "flash"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_fa = attn(input_, freqs_cis)

    rtol = ERROR_RTOL[torch.bfloat16]
    atol = ERROR_ATOL[torch.bfloat16]
    assert output_sdpa.shape == output_fa.shape
    torch.testing.assert_close(output_sdpa, output_fa, rtol=rtol, atol=atol)
