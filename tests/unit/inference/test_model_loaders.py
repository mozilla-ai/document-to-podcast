from typing import Dict, Any

import pytest
from llama_cpp import Llama

from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_tts_model,
)
from outetts.version.v1.interface import InterfaceGGUF


def test_load_llama_cpp_model():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    assert isinstance(model, Llama)
    # we set n_ctx=0 to indicate that we want to use the model's default context
    assert model.n_ctx() == 2048


@pytest.mark.parametrize(
    "model_id, expected_model_type, expected_custom_args",
    [
        ["OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf", InterfaceGGUF, {}],
    ],
)
def test_load_tts_model(
    model_id: str,
    expected_model_type: InterfaceGGUF,
    expected_custom_args: Dict[str, Any],
) -> None:
    model = load_tts_model(model_id)
    assert isinstance(model.model, expected_model_type)
    assert model.model_id == model_id
    for (k, v), (e_k, e_v) in zip(
        model.custom_args.items(), expected_custom_args.items()
    ):
        assert k == e_k
        assert isinstance(v, e_v)
