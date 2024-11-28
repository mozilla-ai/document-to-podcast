from llama_cpp import Llama

from document_to_podcast.inference.model_loaders import load_llama_cpp_model


def test_load_llama_cpp_model():
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    assert isinstance(model, Llama)
    # we set n_ctx=0 to indicate that we want to use the model's default context
    assert model.n_ctx() == 2048
