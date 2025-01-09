from outetts.version.v1.interface import InterfaceGGUF

from document_to_podcast.inference.model_loaders import TTSModel
from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech_oute(mocker):
    model = mocker.MagicMock(spec_set=InterfaceGGUF)
    tts_model = TTSModel(
        model=model,
        model_id="OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf",
        sample_rate=0,
        custom_args={},
    )
    text_to_speech(
        input_text="Hello?",
        model=tts_model,
        voice_profile="female_1",
    )

    model.load_default_speaker.assert_called_with(name=mocker.ANY)
    model.generate.assert_called_with(
        text=mocker.ANY,
        temperature=mocker.ANY,
        repetition_penalty=mocker.ANY,
        max_length=mocker.ANY,
        speaker=mocker.ANY,
    )
