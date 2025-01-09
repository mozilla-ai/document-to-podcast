from huggingface_hub import hf_hub_download
import torch
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF
from dataclasses import dataclass, field


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
        n_ctx=0,  # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
        verbose=False,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
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
        model (InterfaceGGUF): A TTS model that has a .generate() method or similar
            that takes text as input, and returns an audio in the form of a numpy array.
        model_id (str): The model's identifier string.
        sample_rate (int): The sample rate of the audio, required for properly saving the audio to a file.
        custom_args (dict): Any model-specific arguments that a TTS model might require, e.g. tokenizer.
    """

    model: InterfaceGGUF
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
        model_path=local_path,
        language=kwargs.pop("language", "en"),
        n_gpu_layers=-1 if torch.cuda.is_available else 0,
        additional_model_config={"verbose": False},
    )
    model = InterfaceGGUF(model_version=model_version, cfg=model_config)

    return TTSModel(
        model=model, model_id=model_id, sample_rate=model.audio_codec.sr, custom_args={}
    )


TTS_LOADERS = {
    # To add support for your model, add it here in the format {model_id} : _load_function
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": _load_oute_tts,
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": _load_oute_tts,
}


def load_tts_model(model_id: str, **kwargs) -> TTSModel:
    return TTS_LOADERS[model_id](model_id, **kwargs)
