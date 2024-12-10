from pathlib import Path

import pytest

from document_to_podcast.inference.model_loaders import load_outetts_model
from document_to_podcast.podcast_maker.config import (
    PodcastConfig,
    SpeakerConfig,
)


@pytest.fixture(scope="session")
def example_data():
    return Path(__file__).parent.parent / "example_data"


@pytest.fixture()
def tts_prompt():
    return "Wow what a great unit test this is!"


@pytest.fixture()
def podcast_script():
    return '{"Speaker 1": "Welcome to our podcast.", "Speaker 2": "It\'s great to be here!"}'


@pytest.fixture()
def podcast_config():
    model = load_outetts_model(
        "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf", "en", "cpu"
    )
    speaker_1 = SpeakerConfig(
        model=model,
        speaker_id="1",
        speaker_profile="female_1",
    )
    speaker_2 = SpeakerConfig(
        model=model,
        speaker_id="2",
        speaker_profile="male_1",
    )
    speakers = {s.speaker_id: s for s in [speaker_1, speaker_2]}
    return PodcastConfig(speakers=speakers)
