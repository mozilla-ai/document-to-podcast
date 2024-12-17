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


def _text_to_speech_bark(
    input_test: str, model: BarkModel, processor: BarkProcessor, voice_profile: str
) -> np.ndarray:
    inputs = processor(input_test, voice_preset=voice_profile)

    generation = model.generate(**inputs)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    voice_profile: str,
) -> np.ndarray:
    input_ids = tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler_multi(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    voice_profile: str,
    description_tokenizer: PreTrainedTokenizerBase = None,
) -> np.ndarray:
    input_ids = description_tokenizer(voice_profile, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def _text_to_speech_parler_indic(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    voice_profile: str,
    description_tokenizer: PreTrainedTokenizerBase = None,
) -> np.ndarray:
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
