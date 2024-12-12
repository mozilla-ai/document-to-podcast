from typing import Tuple

from llama_cpp import Llama
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_llama_cpp_model(model_id: str, device: str = "cpu") -> Llama:
    """
    Loads the given model_id using Llama.from_pretrained.

    Examples:
        >>> model = load_llama_cpp_model(
            "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf", "cpu")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.
        device (str): The device to load the model on, such as "cuda:0" or "cpu".

    Returns:
        Llama: The loaded model.
    """
    org, repo, filename = model_id.split("/")
    model = Llama.from_pretrained(
        repo_id=f"{org}/{repo}",
        filename=filename,
        # -1 means offload all layers to GPU, but you can also define to have some in CPU, some in GPU
        n_gpu_layers=0 if device == "cpu" else -1,
        # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
        n_ctx=0,
        verbose=True,
    )
    return model


def load_parler_tts_model_and_tokenizer(
    model_id: str, device: str = "cpu"
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the given model_id using parler_tts.from_pretrained.

    Examples:
        >>> model, tokenizer = load_parler_tts_model_and_tokenizer("parler-tts/parler-tts-mini-v1", "cpu")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.
        device (str): The device to load the model on, such as "cuda:0" or "cpu".

    Returns:
        PreTrainedModel: The loaded model.
    """
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer
