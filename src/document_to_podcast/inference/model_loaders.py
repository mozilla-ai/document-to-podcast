from typing import Union
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from outetts import GGUFModelConfig_v1, InterfaceGGUF
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoProcessor,
    BarkModel,
)
import numpy as np

from document_to_podcast.inference.text_to_speech import (
    _text_to_speech_oute,
    _text_to_speech_bark,
    _text_to_speech_parler,
    _text_to_speech_parler_multi,
    _text_to_speech_parler_indic,
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


class TTSInterface:
    """
    The purpose of this class is to provide a unified interface for all the TTS models supported.
    Specifically, different TTS model families have different peculiarities, for example, the bark model needs a
    BarkProcessor, the parler models need their own tokenizer, etc. This wrapper takes care of this complexity so that
    the user doesn't have to deal with it.
    """

    def __init__(
        self,
        model: Union[InterfaceGGUF, BarkModel, PreTrainedModel],
        model_id: str,
        sample_rate: int,
        **kwargs,
    ):
        self.model = model
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.kwargs = kwargs

    def text_to_speech(
        self, input_text: str, voice_profile: str, **kwargs
    ) -> np.ndarray:
        """
        Generates a speech waveform from a text input using a pre-trained text-to-speech (TTS) model.

        Args:
            input_text (str): The text to convert to speech.
            voice_profile (str): Depending on the selected TTS model it should either be
                - a pre-defined ID for the Oute models (e.g. "female_1")
                more info here https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers
                - a pre-defined ID for the Bark model (e.g. "v2/en_speaker_0")
                more info here https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
                - a natural description of the voice profile using a pre-defined name for the Parler model (e.g. Laura's voice is calm)
                more info here https://github.com/huggingface/parler-tts?tab=readme-ov-file#-using-a-specific-speaker
                for the multilingual model: https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1
                for the indic model: https://huggingface.co/ai4bharat/indic-parler-tts
        Returns:
            numpy array: The waveform of the speech as a 2D numpy array
        """
        return SUPPORTED_TTS_MODELS[self.model_id][1](
            input_text, self.model, voice_profile, **self.kwargs | kwargs
        )


def load_tts_model(model_id: str, **kwargs) -> TTSInterface:
    """

    Args:
        model_id:
        outetts_language:

    Returns:

    """
    return SUPPORTED_TTS_MODELS[model_id][0](model_id, **kwargs)


def _load_oute_tts(model_id: str, language: str = "en") -> TTSInterface:
    """
    Loads the given model_id using the OuteTTS interface. For more info: https://github.com/edwko/OuteTTS

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{org}/{repo}/{filename}`.
        language (str): Supported languages in 0.2-500M: en, zh, ja, ko.
    Returns:
        TTSInterface: The loaded model using the TTSModelWrapper.
    """
    model_version = model_id.split("-")[1]

    org, repo, filename = model_id.split("/")
    local_path = hf_hub_download(repo_id=f"{org}/{repo}", filename=filename)
    model_config = GGUFModelConfig_v1(model_path=local_path, language=language)
    model = InterfaceGGUF(model_version=model_version, cfg=model_config)

    return TTSInterface(
        model=model, model_id=model_id, sample_rate=model.audio_codec.sr
    )


def _load_bark_tts(model_id: str, **kwargs) -> TTSInterface:
    """
    Loads the given model_id and its required processor. For more info: https://github.com/suno-ai/bark

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        TTSInterface: The loaded model with its required processor using the TTSModelWrapper.
    """

    processor = AutoProcessor.from_pretrained(model_id)
    model = BarkModel.from_pretrained(model_id)

    return TTSInterface(
        model=model, model_id=model_id, sample_rate=24_000, bark_processor=processor
    )


def _load_parler_tts(model_id: str, **kwargs) -> TTSInterface:
    """
    Loads the given model_id using parler_tts.from_pretrained. For more info: https://github.com/huggingface/parler-tts

    Args:
        model_id (str): The model id to load.
            Format is expected to be `{repo}/{filename}`.

    Returns:
        TTSInterface: The loaded model with its required tokenizer for the input. For the multilingual models we also
        load another tokenizer for the description
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

    return TTSInterface(
        model=model,
        model_id=model_id,
        sample_rate=model.config.sampling_rate,
        parler_tokenizer=tokenizer,
        parler_description_tokenizer=description_tokenizer,
    )


SUPPORTED_TTS_MODELS = {
    # To add support for your model, add it here in the format {model_id} : [_load_function, _text_to_speech_function]
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": [
        _load_oute_tts,
        _text_to_speech_oute,
    ],
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": [
        _load_oute_tts,
        _text_to_speech_oute,
    ],
    "suno/bark": [_load_bark_tts, _text_to_speech_bark],
    "parler-tts/parler-tts-large-v1": [_load_parler_tts, _text_to_speech_parler],
    "parler-tts/parler-tts-mini-v1": [_load_parler_tts, _text_to_speech_parler],
    "parler-tts/parler-tts-mini-v1.1": [_load_parler_tts, _text_to_speech_parler],
    "parler-tts/parler-tts-mini-multilingual-v1.1": [
        _load_parler_tts,
        _text_to_speech_parler_multi,
    ],
    "ai4bharat/indic-parler-tts": [_load_parler_tts, _text_to_speech_parler_indic],
}
