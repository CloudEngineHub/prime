import torch
from zeroband.checkpoint import get_sendable_opt_state, load_sendable_opt_state


def test_load_state_dict():
    state_dict_to_send = {
        "step": 0,
        "world": "karl is having his best life",
        "optim_sates": torch.ones(10),
        "nested_data": {"foo": "bar", "tensor": torch.ones(10)},
    }

    state_dict_copy = {
        "step": 0,
        "world": "karl is having his best life",
        "optim_sates": torch.ones(10),
        "nested_data": {"foo": "bar", "tensor": torch.ones(10)},
    }

    non_tensored_state_send, tensors_send = get_sendable_opt_state(state_dict_to_send)

    assert isinstance(non_tensored_state_send["optim_sates"], str)
    assert non_tensored_state_send["optim_sates"].startswith("zeroband_tensor")

    print(len(tensors_send))
    print(non_tensored_state_send)
    load_sendable_opt_state(tensors_send, non_tensored_state_send)

    assert (state_dict_to_send["optim_sates"] == state_dict_copy["optim_sates"]).all()
    assert id(state_dict_to_send["optim_sates"]) != id(state_dict_copy["optim_sates"])

    assert (state_dict_to_send["nested_data"]["tensor"] == state_dict_copy["nested_data"]["tensor"]).all()
    assert id(state_dict_to_send["nested_data"]["tensor"]) != id(state_dict_copy["nested_data"]["tensor"])

    assert state_dict_to_send["step"] == state_dict_copy["step"]
    assert state_dict_to_send["world"] == state_dict_copy["world"]
    assert state_dict_to_send["nested_data"]["foo"] == state_dict_copy["nested_data"]["foo"]
