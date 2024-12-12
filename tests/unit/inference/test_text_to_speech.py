from document_to_podcast.inference.text_to_speech import text_to_speech


def test_text_to_speech(mocker):
    model = mocker.MagicMock()
    text_to_speech(
        "Hello?",
        model=model,
        voice_profile="male_1",
    )
    model.generate.assert_called_with(input_ids=mocker.ANY, prompt_input_ids=mocker.ANY)
