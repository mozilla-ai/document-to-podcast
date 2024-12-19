from typing import Union
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoProcessor,
    BarkModel,
)


def load_llama_cpp_model(model_id: str) -> Llama:
    """
    Loads the given model_id using Llama.from_pretrained.

    Examples:
        >>> model = load_llama_cpp_model("allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf")

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


@dataclass
class TTSModel:
    """
    The purpose of this class is to provide a unified interface for all the TTS models supported.
    Specifically, different TTS model families have different peculiarities, for example, the bark models need a
    BarkProcessor, the parler models need their own tokenizer, etc. This wrapper takes care of this complexity so that
    the user doesn't have to deal with it.

    Args:
        model (Union[InterfaceGGUF, BarkModel, PreTrainedModel]): A TTS model that has a .generate() method or similar
            that takes text as input, and returns an audio in the form of a numpy array.
        model_id (str): The model's identifier string.
        sample_rate (int): The sample rate of the audio, required for properly saving the audio to a file.
        custom_args (dict): Any model-specific arguments that a TTS model might require, e.g. tokenizer.
    """

    model: Union[InterfaceGGUF, BarkModel, PreTrainedModel]
    model_id: str
    sample_rate: int
    custom_args: field(default_factory=dict)


def _load_oute_tts(model_id: str, **kwargs) -> TTSModel:
    """
    Loads the given model_id using the OuteTTS interface. For more info: https://github.com/edwko/OuteTTS

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.
        language (str): Supported languages in 0.2-500M: en, zh, ja, ko.
    Returns:
        TTSModel: The loaded model using the TTSModel wrapper.
    """
    model_version = model_id.split("-")[1]

    org, repo, filename = model_id.split("/")
    local_path = hf_hub_download(repo_id=f"{org}/{repo}", filename=filename)
    model_config = GGUFModelConfig_v1(
        model_path=local_path, language=kwargs.pop("language", "en")
    )
    model = InterfaceGGUF(model_version=model_version, cfg=model_config)

    return TTSModel(model=model, model_id=model_id, sample_rate=model.audio_codec.sr)


def _load_bark_tts(model_id: str, **kwargs) -> TTSModel:
    """
    Loads the given model_id and its required processor. For more info: https://github.com/suno-ai/bark

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.
    Returns:
        TTSModel: The loaded model with its required processor using the TTSModel.
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = BarkModel.from_pretrained(model_id)

    return TTSModel(
        model=model,
        model_id=model_id,
        sample_rate=24_000,
        custom_args={"processor": processor},
    )


def _load_parler_tts(model_id: str, **kwargs) -> TTSModel:
    """
    Loads the given model_id using parler_tts.from_pretrained. For more info: https://github.com/huggingface/parler-tts

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        TTSModel: The loaded model with its required tokenizer for the input.
    """
    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return TTSModel(
        model=model,
        model_id=model_id,
        sample_rate=model.config.sampling_rate,
        custom_args={
            "tokenizer": tokenizer,
        },
    )


def _load_parler_tts_multi(model_id: str, **kwargs) -> TTSModel:
    """
    Loads the given model_id using parler_tts.from_pretrained. For more info: https://github.com/huggingface/parler-tts

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        TTSModel: The loaded model with its required tokenizer for the input text and
            another tokenizer for the description.
    """

    from parler_tts import ParlerTTSForConditionalGeneration

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    return TTSModel(
        model=model,
        model_id=model_id,
        sample_rate=model.config.sampling_rate,
        custom_args={
            "tokenizer": tokenizer,
            "description_tokenizer": description_tokenizer,
        },
    )


TTS_LOADERS = {
    # To add support for your model, add it here in the format {model_id} : _load_function
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": _load_oute_tts,
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": _load_oute_tts,
    "suno/bark": _load_bark_tts,
    "suno/bark-small": _load_bark_tts,
    "parler-tts/parler-tts-large-v1": _load_parler_tts,
    "parler-tts/parler-tts-mini-v1": _load_parler_tts,
    "parler-tts/parler-tts-mini-v1.1": _load_parler_tts,
    "parler-tts/parler-tts-mini-multilingual-v1.1": _load_parler_tts_multi,
    "ai4bharat/indic-parler-tts": _load_parler_tts_multi,
}


def load_tts_model(model_id: str, **kwargs) -> TTSModel:
    return TTS_LOADERS[model_id](model_id, **kwargs)
