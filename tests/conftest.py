from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def example_data():
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture()
def tts_prompt():
    return "Wow what a great unit test this is!"


@pytest.fixture()
def podcast_script():
    return '{"Speaker 1": "Welcome to our podcast.", "Speaker 2": "It\'s great to be here!"}'
