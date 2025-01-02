from outetts.version.v1.interface import InterfaceGGUF
from transformers import PreTrainedModel

from document_to_podcast.inference.model_loaders import TTSModel
from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech_oute(mocker):
    model = mocker.MagicMock(spec_set=InterfaceGGUF)
    tts_model = TTSModel(
        model=model,
        model_id="OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-Q2_K.gguf",
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


def test_text_to_speech_bark(mocker):
    model = mocker.MagicMock(spec_set=PreTrainedModel)
    processor = mocker.MagicMock()

    tts_model = TTSModel(
        model=model,
        model_id="suno/bark",
        sample_rate=0,
        custom_args={"processor": processor},
    )
    text_to_speech(
        input_text="Hello?",
        model=tts_model,
        voice_profile="v2/en_speaker_0",
    )
    processor.assert_has_calls(
        [
            mocker.call("Hello?", voice_preset="v2/en_speaker_0"),
        ]
    )
    model.generate.assert_called_with()


def test_text_to_speech_parler(mocker):
    model = mocker.MagicMock(spec_set=PreTrainedModel)
    tokenizer = mocker.MagicMock()

    tts_model = TTSModel(
        model=model,
        model_id="parler-tts/parler-tts-mini-v1.1",
        sample_rate=0,
        custom_args={"tokenizer": tokenizer},
    )
    text_to_speech(
        input_text="Hello?",
        model=tts_model,
        voice_profile="Laura's voice is calm",
    )
    tokenizer.assert_has_calls(
        [
            mocker.call("Laura's voice is calm", return_tensors="pt"),
            mocker.call("Hello?", return_tensors="pt"),
        ]
    )
    model.generate.assert_called_with(input_ids=mocker.ANY, prompt_input_ids=mocker.ANY)


def test_text_to_speech_parler_multi(mocker):
    model = mocker.MagicMock(spec_set=PreTrainedModel)
    tokenizer = mocker.MagicMock()
    description_tokenizer = mocker.MagicMock()

    tts_model = TTSModel(
        model=model,
        model_id="ai4bharat/indic-parler-tts",
        sample_rate=0,
        custom_args={
            "tokenizer": tokenizer,
            "description_tokenizer": description_tokenizer,
        },
    )
    text_to_speech(
        input_text="Hello?",
        model=tts_model,
        voice_profile="Laura's voice is calm",
    )
    description_tokenizer.assert_has_calls(
        [
            mocker.call("Laura's voice is calm", return_tensors="pt"),
        ]
    )
    tokenizer.assert_has_calls(
        [
            mocker.call("Hello?", return_tensors="pt"),
        ]
    )
    model.generate.assert_called_with(input_ids=mocker.ANY, prompt_input_ids=mocker.ANY)
