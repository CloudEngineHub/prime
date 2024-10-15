import os
import pytest
import torch
from zeroband.comms import ElasticDeviceMesh
from zeroband.utils.state_dict_send_recv import (
    _get_sendable_state_dict,
    _load_sendable_state_dict,
    recv_state_dict,
    send_state_dict,
)
import multiprocessing as mp


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

    non_tensored_state_send, tensors_send = _get_sendable_state_dict(state_dict_to_send)

    assert isinstance(non_tensored_state_send["optim_sates"], str)
    assert non_tensored_state_send["optim_sates"].startswith("zeroband_tensor")

    print(len(tensors_send))
    print(non_tensored_state_send)
    _load_sendable_state_dict(tensors_send, non_tensored_state_send)

    assert (state_dict_to_send["optim_sates"] == state_dict_copy["optim_sates"]).all()
    assert id(state_dict_to_send["optim_sates"]) != id(state_dict_copy["optim_sates"])

    assert (state_dict_to_send["nested_data"]["tensor"] == state_dict_copy["nested_data"]["tensor"]).all()
    assert id(state_dict_to_send["nested_data"]["tensor"]) != id(state_dict_copy["nested_data"]["tensor"])

    assert state_dict_to_send["step"] == state_dict_copy["step"]
    assert state_dict_to_send["world"] == state_dict_copy["world"]
    assert state_dict_to_send["nested_data"]["foo"] == state_dict_copy["nested_data"]["foo"]


@pytest.mark.skip(reason="hang")
@pytest.mark.parametrize("world_size", [2])
def test_send_recv_state_dict(world_size: int, random_available_port: int, mock_env):
    def foo(**kwargs):
        with mock_env(**kwargs):
            edm = ElasticDeviceMesh()

            state_dict_to_send = {
                "step": 0,
                "world": "karl is having his best life",
                "optim_sates": torch.ones(10),
                "nested_data": {"foo": "bar", "tensor": torch.ones(10)},
            }

            state_dict_to_recv = {
                "step": 10,
                "world": "karl is in holiday",
                "optim_sates": torch.zeros(10),
                "nested_data": {"foo": "barman", "tensor": torch.zeros(10)},
            }

            rank = int(os.environ.get("RANK"))

            if rank == 0:
                send_state_dict(state_dict_to_send, 1, world_size)
            else:
                state_dict = recv_state_dict(pg=edm.global_pg, rank=0, world_size=world_size)

                assert (state_dict["optim_sates"] == state_dict_to_recv["optim_sates"]).all()
                assert id(state_dict["optim_sates"]) != id(state_dict_to_recv["optim_sates"])

                assert (state_dict["nested_data"]["tensor"] == state_dict_to_recv["nested_data"]["tensor"]).all()
                assert id(state_dict["nested_data"]["tensor"]) != id(state_dict_to_recv["nested_data"]["tensor"])

                assert state_dict["step"] == state_dict_to_recv["step"]
                assert state_dict["world"] == state_dict_to_recv["world"]
                assert state_dict["nested_data"]["foo"] == state_dict_to_recv["nested_data"]["foo"]

            del edm

    processes = []
    for rank in range(world_size):
        processes.append(
            mp.Process(
                target=foo,
                kwargs={
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(random_available_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": str(rank),
                    "LOCAL_WORLD_SIZE": str(world_size),
                    "ZERO_BAND_LOG_LEVEL": "DEBUG",
                },
            )
        )
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        if p.exitcode != 0:
            pytest.fail(f"Process {p.pid} failed with exit code {p.exitcode}")
