from typing import Generator

from kokoro import KPipeline

from document_to_podcast.inference.model_loaders import TTSModel
from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech_kokoro(mocker):
    model = mocker.MagicMock(spec_set=KPipeline)
    generator = mocker.MagicMock(spec_set=Generator)
    tts_model = TTSModel(
        model=model,
        model_id="hexgrad/Kokoro-82M",
        sample_rate=0,
        custom_args={},
    )
    text_to_speech(
        input_text="Hello?",
        model=tts_model,
        voice_profile="af_sarah",
    )

    model.__call__.assert_called_with(
        text=mocker.ANY,
        voice=mocker.ANY,
    )
    generator.__next__.assert_called_with()
