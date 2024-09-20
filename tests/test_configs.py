"""
Tests all of the config file. usefull to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

import os
from zeroband.train import Config
import pytest
import tomli

config_file_names = [file for file in os.listdir("configs") if file.endswith(".toml")]

@pytest.mark.parametrize("config_file_name", config_file_names)
def test_load_config(config_file_name):
    with open(f"configs/{config_file_name}", "rb") as f:
        content = tomli.load(f)
    config = Config(**content)
    assert config is not None

