import json
from typing import Iterator

import numpy as np
import pytest

from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_outetts_model,
    load_parler_tts_model_and_tokenizer,
)
from document_to_podcast.inference.text_to_speech import text_to_speech
from document_to_podcast.inference.text_to_text import text_to_text, text_to_text_stream


def test_model_load_and_inference_text_to_text():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    result = text_to_text(
        "Answer to: What is the capital of France?",
        model=model,
        system_prompt="",
    )
    assert isinstance(result, str)
    assert json.loads(result)


def test_model_load_and_inference_text_to_text_no_json():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    result = text_to_text(
        "What is the capital of France?",
        model=model,
        system_prompt="",
        return_json=False,
        stop=".",
    )
    assert isinstance(result, str)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)
    assert result.startswith("The capital of France is Paris")


def test_model_load_and_inference_text_to_text_stream_no_json():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    result = text_to_text_stream(
        "What is the capital of France?",
        model=model,
        system_prompt="",
        return_json=False,
        stop=".",
    )
    assert isinstance(result, Iterator)
    result = "".join(result)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result)
    assert result.startswith("The capital of France is Paris")


def test_model_load_and_inference_text_to_speech_oute():
    model = load_outetts_model(
        "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf", "en", "cpu"
    )

    speech = text_to_speech(
        input_text="What a pretty unit test this is!",
        model=model,
        speaker_profile="female_1",
    )

    assert isinstance(speech, np.ndarray)
    assert speech.size > 1


def test_model_load_and_inference_text_to_speech_parler():
    model, tokenizer = load_parler_tts_model_and_tokenizer(
        "parler-tts/parler-tts-mini-v1", "cpu"
    )

    speech = text_to_speech(
        input_text="What a pretty unit test this is!",
        model=model,
        tokenizer=tokenizer,
        speaker_profile="Laura's voice is exciting and fast in delivery with very clear audio and no background noise.",
    )

    assert isinstance(speech, np.ndarray)
    assert speech.size > 1
