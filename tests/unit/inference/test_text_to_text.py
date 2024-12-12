import pytest

from document_to_podcast.inference.text_to_text import text_to_text, text_to_text_stream


def test_text_to_text(mocker):
    model = mocker.MagicMock()
    text_to_text(
        "Hello?",
        model=model,
        system_prompt="You are a helpful assistant.",
    )
    model.create_chat_completion.assert_called_with(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello?"},
        ],
        response_format={"type": "json_object"},
        stop=[],
        stream=False,
    )


def test_text_to_text_no_return_json(mocker):
    model = mocker.MagicMock()
    text_to_text(
        "Hello?",
        model=model,
        system_prompt="You are a helpful assistant.",
        return_json=False,
    )
    model.create_chat_completion.assert_called_with(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello?"},
        ],
        response_format=None,
        stop=[],
        stream=False,
    )


def test_text_to_text_stream_no_return_json(mocker):
    model = mocker.MagicMock()
    iterator = text_to_text_stream(
        "Hello?",
        model=model,
        system_prompt="You are a helpful assistant.",
        return_json=False,
    )
    with pytest.raises(StopIteration):
        next(iterator)
    model.create_chat_completion.assert_called_with(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello?"},
        ],
        response_format=None,
        stop=[],
        stream=True,
    )


def test_text_to_text_stream(mocker):
    model = mocker.MagicMock()
    iterator = text_to_text_stream(
        "Hello?",
        model=model,
        system_prompt="You are a helpful assistant.",
    )
    with pytest.raises(StopIteration):
        next(iterator)
    model.create_chat_completion.assert_called_with(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello?"},
        ],
        response_format={"type": "json_object"},
        stop=[],
        stream=True,
    )
