from typing import Tuple
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoProcessor,
    BarkModel,
    BarkProcessor,
)


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


def load_outetts_model(model_id: str, language: str = "en") -> InterfaceGGUF:
    """
    Loads the given model_id using the OuteTTS interface. For more info: https://github.com/edwko/OuteTTS

    Examples:
        >>> model = load_outetts_model("OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf", "en")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.
        language (str): Supported languages in 0.2-500M: en, zh, ja, ko.
    Returns:
        PreTrainedModel: The loaded model.
    """
    model_version = model_id.split("-")[1]

    org, repo, filename = model_id.split("/")
    local_path = hf_hub_download(repo_id=f"{org}/{repo}", filename=filename)
    model_config = GGUFModelConfig_v1(model_path=local_path, language=language)

    return InterfaceGGUF(model_version=model_version, cfg=model_config)


def load_bark_tts(model_id: str) -> Tuple[BarkModel, BarkProcessor]:
    """
    Loads the given model_id and its required processor. For more info: https://github.com/suno-ai/bark

    Examples:
        >>> model, processor = load_bark_tts("suno/bark", "cpu")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        BarkModel: The loaded model.
        BarkProcessor: The loaded model.
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = BarkModel.from_pretrained(model_id)

    return model, processor


def load_parler_tts_model_and_tokenizer(
    model_id: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, PreTrainedTokenizerBase]:
    """
    Loads the given model_id using parler_tts.from_pretrained. For more info: https://github.com/huggingface/parler-tts

    Examples:
        >>> model, tokenizer, _ = load_parler_tts_model_and_tokenizer("parler-tts/parler-tts-mini-v1")

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        PreTrainedModel: The loaded model.
        PreTrainedTokenizer: The loaded tokenizer for the input.
        PreTrainedTokenizer: [Only for the multilingual models] The loaded tokenizer for the description.
    """
    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    description_tokenizer = (
        AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        if model_id == "parler-tts/parler-tts-mini-multilingual-v1.1"
        or model_id == "ai4bharat/indic-parler-tts"
        else None
    )

    return model, tokenizer, description_tokenizer
