from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech_parler(mocker):
    model = mocker.MagicMock()
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
