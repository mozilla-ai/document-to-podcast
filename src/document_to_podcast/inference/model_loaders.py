from typing import Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF
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


def load_outetts_model(
    model_id: str, language: str = "en", device: str = "cpu"
) -> InterfaceGGUF:
    """
    Loads the given model_id using the OuteTTS interface. For more info: https://github.com/edwko/OuteTTS

    Examples:
        >>> model = load_outetts_model("OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf", "en", "cpu")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.
        language (str): Supported languages in 0.2-500M: en, zh, ja, ko.
        device (str): The device to load the model on, such as "cuda:0" or "cpu".

    Returns:
        PreTrainedModel: The loaded model.
    """
    n_layers_on_gpu = 0 if device == "cpu" else -1
    model_version = model_id.split("-")[1]

    org, repo, filename = model_id.split("/")
    local_path = hf_hub_download(repo_id=f"{org}/{repo}", filename=filename)
    model_config = GGUFModelConfig_v1(
        model_path=local_path, language=language, n_gpu_layers=n_layers_on_gpu
    )

    return InterfaceGGUF(model_version=model_version, cfg=model_config)


def load_parler_tts_model_and_tokenizer(
    model_id: str, device: str = "cpu"
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the given model_id using parler_tts.from_pretrained. For more info: https://github.com/huggingface/parler-tts

    Examples:
        >>> model, tokenizer = load_parler_tts_model_and_tokenizer("parler-tts/parler-tts-mini-v1", "cpu")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.
        device (str): The device to load the model on, such as "cuda:0" or "cpu".

    Returns:
        PreTrainedModel: The loaded model.
    """
    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer
