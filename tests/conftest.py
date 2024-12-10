from pathlib import Path

import pytest

from document_to_podcast.inference.model_loaders import (
    load_parler_tts_model_and_tokenizer,
)
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
    speaker_1_description = "Laura's voice is exciting and fast in delivery with very clear audio and no background noise."
    speaker_2_description = (
        "Jon's voice is calm with very clear audio and no background noise."
    )

    model, tokenizer = load_parler_tts_model_and_tokenizer(
        "parler-tts/parler-tts-mini-v1", "cpu"
    )
    speaker_1 = SpeakerConfig(
        model=model,
        speaker_id="1",
        tokenizer=tokenizer,
        speaker_profile=speaker_1_description,
    )
    speaker_2 = SpeakerConfig(
        model=model,
        speaker_id="2",
        tokenizer=tokenizer,
        speaker_profile=speaker_2_description,
    )
    speakers = {s.speaker_id: s for s in [speaker_1, speaker_2]}
    return PodcastConfig(speakers=speakers)
