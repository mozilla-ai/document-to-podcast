import torch
from kokoro import KPipeline
from llama_cpp import Llama
from dataclasses import dataclass, field


def load_llama_cpp_model(model_id: str) -> Llama:
    """
    Loads the given model_id using Llama.from_pretrained.

    Examples:
        >>> model = load_llama_cpp_model("bartowski/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-f16.gguf")

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
        model: A TTS model that has a .generate() method or similar
            that takes text as input, and returns an audio in the form of a numpy array.
        model_id (str): The model's identifier string.
        sample_rate (int): The sample rate of the audio, required for properly saving the audio to a file.
        custom_args (dict): Any model-specific arguments that a TTS model might require, e.g. tokenizer.
    """

    model: KPipeline
    model_id: str
    sample_rate: int
    custom_args: field(default_factory=dict)


def _load_kokoro_tts(model_id: str, **kwargs) -> TTSModel:
    """
    Loads the kokoro model using the KPipeline from the package https://github.com/hexgrad/kokoro

    Args:
        model_id (str): Identifier for a specific model. Kokoro currently only supports one model.
        lang_code (str): In order to initialize model, we need to set the language used for generation. For example:
            🇪🇸 'e' => Spanish es
            🇫🇷 'f' => French fr-fr
            🇮🇳 'h' => Hindi hi
            🇮🇹 'i' => Italian it
            🇧🇷 'p' => Brazilian Portuguese pt-br
            🇺🇸 'a' => American English
            🇬🇧 'b' => British English
            🇯🇵 'j' => Japanese: you will need to also pip install misaki[ja]
            🇨🇳 'z' => Mandarin Chinese: you will need to also pip install misaki[zh]
    Returns:
        TTSModel: The loaded model using the TTSModel wrapper.
    """
    from kokoro import KPipeline

    # If language code not supplied, assume British English
    pipeline = KPipeline(lang_code=kwargs.pop("lang_code", "b"))
    return TTSModel(
        model=pipeline,
        model_id=model_id,
        sample_rate=24000,  # Kokoro's default sample rate
        custom_args={},
    )


TTS_LOADERS = {
    # To add support for your model, add it here in the format {model_id} : _load_function
    "hexgrad/Kokoro-82M": _load_kokoro_tts,
}


def load_tts_model(model_id: str, **kwargs) -> TTSModel:
    return TTS_LOADERS[model_id](model_id, **kwargs)
