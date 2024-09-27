import torch
import hashlib

TENSOR_SIG_SAMPLE_SIZE = 100


def get_tensor_signature(a: torch.Tensor | torch.nn.Parameter) -> str:
    """
    Get the tensor signature
    """
    while isinstance(a, torch.nn.Parameter):
        a = a.data
    if a.numel() < TENSOR_SIG_SAMPLE_SIZE:
        b = a.as_strided(size=(a.numel(),), stride=(1,))
    else:
        step_size = a.numel() // TENSOR_SIG_SAMPLE_SIZE
        b = a.as_strided(size=(TENSOR_SIG_SAMPLE_SIZE,), stride=(step_size,))
    element_str = "".join([f"{x:.3e}" for x in b])
    element_hash = hashlib.md5(element_str.encode("utf-8")).hexdigest()
    return f"{a.dtype}{a.shape}{a.stride()}<{element_hash}>"


def get_module_signature(module: torch.nn.Module, compress: bool = True) -> str:
    """
    Get the module signature
    """
    state_dict_sig = {name: get_tensor_signature(param) for name, param in module.named_parameters()}
    if compress:
        return hashlib.md5(str(state_dict_sig).encode("utf-8")).hexdigest()
    else:
        return "\n".join(f"{name}: {sig}" for name, sig in state_dict_sig.items())
