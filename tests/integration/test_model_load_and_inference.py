import json
from typing import Iterator

import pytest

from opennotebookllm.inference.model_loaders import load_llama_cpp_model
from opennotebookllm.inference.text_to_text import text_to_text, text_to_text_stream


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
