from typing import Tuple

from llama_cpp import Llama
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_llama_cpp_model(
    model_id: str,
) -> Llama:
    """
    Loads the given model_id using Llama.from_pretrained.

    Examples:
        >>> model = load_llama_cpp_model(
            "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.

    Returns:
        Llama: The loaded model.
    """
    org, repo, filename = model_id.split("/")
    model = Llama.from_pretrained(
        repo_id=f"{org}/{repo}",
        filename=filename,
        # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
        n_ctx=0,
        verbose=False,
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
