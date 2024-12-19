import numpy as np
from outetts.version.v1.interface import InterfaceGGUF
from transformers import PreTrainedModel, BarkModel

from document_to_podcast.inference.model_loaders import TTSModel


def _text_to_speech_oute(
    input_text: str,
    model: InterfaceGGUF,
    voice_profile: str,
    **kwargs,
) -> np.ndarray:
    """
    TTS generation function for the Oute TTS model family.
    Args:
        input_text (str): The text to convert to speech.
        model: A model from the Oute TTS family.
        voice_profile: a pre-defined ID for the Oute models (e.g. "female_1")
            more info here https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers
        temperature (float, default = 0.3): Controls the randomness of predictions by scaling the logits.
            Lower values make the output more focused and deterministic, higher values produce more diverse results.
        repetition_penalty (float, default = 1.1): Applies a penalty to tokens that have already been generated,
            reducing the likelihood of repetition and enhancing text variety.
        max_length (int, default = 4096): Defines the maximum number of tokens for the generated text sequence.

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
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
    TTS generation function for the Bark model family.
    Args:
        input_text (str): The text to convert to speech.
        model: A model from the Bark family.
        voice_profile: a pre-defined ID for the Bark model (e.g. "v2/en_speaker_0")
            more info here https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
        processor: Required BarkProcessor to prepare the input text for the Bark model

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    processor = kwargs.get("processor")
    if processor is None:
        raise ValueError("Bark model requires a processor")

    inputs = processor(input_test, voice_preset=voice_profile)

    generation = model.generate(**inputs)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler(
    input_text: str, model: PreTrainedModel, voice_profile: str, **kwargs
) -> np.ndarray:
    """
    TTS generation function for the Parler TTS model family.
    Args:
        input_text (str): The text to convert to speech.
        model: A model from the Parler TTS family.
        voice_profile: a natural description of the voice profile using a pre-defined name for the Parler model (e.g. Laura's voice is calm)
            more info here https://github.com/huggingface/parler-tts?tab=readme-ov-file#-using-a-specific-speaker
        tokenizer (PreTrainedTokenizer): Required PreTrainedTokenizer to tokenize the input text.

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
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
    TTS generation function for the Parler TTS multilingual model family.
    Args:
        input_text (str): The text to convert to speech.
        model: A model from the Parler TTS multilingual family.
        voice_profile: a natural description of the voice profile using a pre-defined name for the Parler model (e.g. Laura's voice is calm)
            more info here https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1
            for the indic version https://huggingface.co/ai4bharat/indic-parler-tts
        tokenizer (PreTrainedTokenizer): Required PreTrainedTokenizer to tokenize the input text.
        description_tokenizer (PreTrainedTokenizer): Required PreTrainedTokenizer to tokenize the description text.

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    tokenizer = kwargs.get("tokenizer")
    if tokenizer is None:
        raise ValueError("Parler model requires a tokenizer")

    description_tokenizer = kwargs.get("description_tokenizer")
    if description_tokenizer is None:
        raise ValueError("Parler multilingual model requires a description tokenizer")

    input_ids = description_tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()
    return waveform


TTS_INFERENCE = {
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-Q2_K.gguf": _text_to_speech_oute,
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": _text_to_speech_oute,
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": _text_to_speech_oute,
    "suno/bark": _text_to_speech_bark,
    "parler-tts/parler-tts-large-v1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-v1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-v1.1": _text_to_speech_parler,
    "parler-tts/parler-tts-mini-multilingual-v1.1": _text_to_speech_parler_multi,
    "ai4bharat/indic-parler-tts": _text_to_speech_parler_multi,
}


def text_to_speech(input_text: str, model: TTSModel, voice_profile: str) -> np.ndarray:
    return TTS_INFERENCE[model.model_id](
        input_text, model.model, voice_profile, **model.custom_args
    )
