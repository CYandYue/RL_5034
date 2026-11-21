import pytest

from hvac_env import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()
