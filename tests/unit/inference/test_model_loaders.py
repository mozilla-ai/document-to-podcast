from llama_cpp import Llama

from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_outetts_model,
)
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from outetts.version.v1.interface import InterfaceGGUF

from document_to_podcast.inference.model_loaders import (
    load_parler_tts_model_and_tokenizer,
)


def test_load_llama_cpp_model():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    assert isinstance(model, Llama)
    # we set n_ctx=0 to indicate that we want to use the model's default context
    assert model.n_ctx() == 2048


def test_load_outetts_model():
    model = load_outetts_model(
        "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-Q2_K.gguf"
    )
    assert isinstance(model, InterfaceGGUF)


def test_load_parler_tts_model_and_tokenizer():
    model, tokenizer = load_parler_tts_model_and_tokenizer(
        "parler-tts/parler-tts-mini-v1"
    )
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
