from outetts.version.v1.interface import InterfaceGGUF
from transformers import PreTrainedModel

from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech_oute(mocker):
    model = mocker.MagicMock(spec_set=InterfaceGGUF)
    text_to_speech(
        "Hello?",
        model=model,
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


def test_text_to_speech_parler(mocker):
    model = mocker.MagicMock(spec_set=PreTrainedModel)
    tokenizer = mocker.MagicMock()
    text_to_speech(
        "Hello?",
        model=model,
        tokenizer=tokenizer,
        voice_profile="default",
    )
    tokenizer.assert_has_calls(
        [
            mocker.call("default", return_tensors="pt"),
            mocker.call("Hello?", return_tensors="pt"),
        ]
    )
    model.generate.assert_called_with(input_ids=mocker.ANY, prompt_input_ids=mocker.ANY)
