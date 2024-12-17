from typing import Union

import numpy as np
from outetts.version.v1.interface import InterfaceGGUF
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BarkModel,
    BarkProcessor,
)


def _text_to_speech_oute(
    input_text: str,
    model: InterfaceGGUF,
    voice_profile: str,
    temperature: float = 0.3,
) -> np.ndarray:
    speaker = model.load_default_speaker(name=voice_profile)

    output = model.generate(
        text=input_text,
        temperature=temperature,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=speaker,
    )

    output_as_np = output.audio.cpu().detach().numpy().squeeze()
    return output_as_np


def _text_to_speech__bark(
    input_test: str, model: BarkModel, processor: BarkProcessor, voice_profile: str
):
    inputs = processor(input_test, voice_preset=voice_profile)

    generation = model.generate(**inputs)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    voice_profile: str,
    description_tokenizer: PreTrainedTokenizerBase = None,
) -> np.ndarray:
    if description_tokenizer:
        input_ids = description_tokenizer(voice_profile, return_tensors="pt").input_ids
    else:
        input_ids = tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def text_to_speech(
    input_text: str,
    model: Union[InterfaceGGUF, PreTrainedModel, BarkModel],
    voice_profile: str,
    processor: Union[BarkProcessor, PreTrainedTokenizerBase] = None,
    description_tokenizer: PreTrainedTokenizerBase = None,
) -> np.ndarray:
    """
    Generates a speech waveform from a text input using a pre-trained text-to-speech (TTS) model.

    Examples:
        >>> waveform = text_to_speech(input_text="Welcome to our amazing podcast", model=model, voice_profile="male_1")

    Args:
        input_text (str): The text to convert to speech.
        model (PreTrainedModel): The model used for generating the waveform.
        voice_profile (str): Depending on the selected TTS model it should either be
            - a pre-defined ID for the Oute models (e.g. "female_1")
            more info here https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers
            - a pre-defined ID for the Bark model (e.g. "v2/en_speaker_0")
            more info here https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
            - a natural description of the voice profile using a pre-defined name for the Parler model (e.g. Laura's voice is calm)
            more info here https://github.com/huggingface/parler-tts?tab=readme-ov-file#-using-a-specific-speaker
            for the multilingual model: https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1
            for the indic model: https://huggingface.co/ai4bharat/indic-parler-tts
        processor (BarkProcessor or PreTrainedTokenizerBase): [Only used for the Bark or Parler models!]
            In bark models, this is an HF processor. In Parler models, this is a pretrained tokenizer.
    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    if isinstance(model, InterfaceGGUF):
        return _text_to_speech_oute(input_text, model, voice_profile)
    elif isinstance(model, BarkModel):
        return _text_to_speech__bark(input_text, model, processor, voice_profile)
    elif isinstance(model, PreTrainedModel):
        return _text_to_speech_parler(
            input_text, model, processor, voice_profile, description_tokenizer
        )
    else:
        raise NotImplementedError
