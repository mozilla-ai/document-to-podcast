import numpy as np
from outetts.version.v1.interface import InterfaceGGUF
from transformers import PreTrainedModel, BarkModel


def _text_to_speech_oute(
    input_text: str,
    model: InterfaceGGUF,
    voice_profile: str,
    **kwargs,
) -> np.ndarray:
    """

    Args:
        input_text:
        model:
        voice_profile:

        temperature: float = 0.3:
        repetition_penalty: float = 1.1
        max_length: int = 4096

    Returns:

    """
    speaker = model.load_default_speaker(name=voice_profile)

    output = model.generate(
        text=input_text,
        temperature=kwargs.pop("temperature", 0.3),
        repetition_penalty=kwargs.pop("repetition_penalty", 1.1),
        max_length=kwargs.pop("max_length", 4096),
        speaker=speaker,
    )

    output_as_np = output.audio.cpu().detach().numpy().squeeze()
    return output_as_np


def _text_to_speech_bark(
    input_test: str, model: BarkModel, voice_profile: str, **kwargs
) -> np.ndarray:
    """

    Args:
        input_test:
        model:
        voice_profile:
        processor: BarkProcessor

    Returns:

    """
    if not (processor := kwargs.pop("processor", None)):
        raise ValueError("Bark model requires a processor")

    inputs = processor(input_test, voice_preset=voice_profile)

    generation = model.generate(**inputs)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler(
    input_text: str, model: PreTrainedModel, voice_profile: str, **kwargs
) -> np.ndarray:
    """

    Args:
        input_text:
        model:
        voice_profile:
        tokenizer: PreTrainedTokenizer

    Returns:

    """

    if not (tokenizer := kwargs.pop("parler_tokenizer", None)):
        raise ValueError("Parler model requires a tokenizer")
    input_ids = tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler_multi(
    input_text: str, model: PreTrainedModel, voice_profile: str, **kwargs
) -> np.ndarray:
    """

    Args:
        input_text:
        model:
        voice_profile:
        tokenizer:
        description_tokenizer:

    Returns:

    """

    if not (tokenizer := kwargs.pop("parler_tokenizer", None)):
        raise ValueError("Parler model requires a tokenizer")

    if not (description_tokenizer := kwargs.pop("parler_description_tokenizer", None)):
        raise ValueError("Parler multilingual model requires a description tokenizer")

    input_ids = description_tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler_indic(
    input_text: str, model: PreTrainedModel, voice_profile: str, **kwargs
) -> np.ndarray:
    """

    Args:
        input_text:
        model:
        voice_profile:
        tokenizer:
        description_tokenizer:

    Returns:

    """
    if not (tokenizer := kwargs.pop("parler_tokenizer", None)):
        raise ValueError("Parler model requires a tokenizer")
    if not (description_tokenizer := kwargs.pop("parler_description_tokenizer", None)):
        raise ValueError("Parler multilingual model requires a description tokenizer")

    input_ids = description_tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(
        input_ids=input_ids.input_ids,
        attention_mask=input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
    )
    waveform = generation.cpu().numpy().squeeze()

    return waveform


TTS_INFERENCE = {
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": _text_to_speech_oute,
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": _text_to_speech_oute,
    "suno/bark": _text_to_speech_bark,
    "parler-tts/parler-tts-large-v1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-v1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-v1.1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-multilingual-v1.1": _text_to_speech_parler_multi,
    "ai4bharat/indic-parler-tts": _text_to_speech_parler_indic,
}


def text_to_speech(input_text, model, voice_profile):
    return TTS_INFERENCE[model.model_id](
        input_text, model.model, voice_profile, **model.custom_args
    )
