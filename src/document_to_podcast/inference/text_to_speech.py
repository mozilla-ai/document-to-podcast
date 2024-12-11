from typing import Union

import numpy as np
from outetts.version.v1.interface import InterfaceGGUF as InterfaceGGUFClass
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def _speech_generation_oute(
    input_text: str,
    model: InterfaceGGUFClass,
    speaker_profile: str,
    temperature: float = 0.3,
) -> np.ndarray:
    speaker = model.load_default_speaker(name=speaker_profile)

    output = model.generate(
        text=input_text,
        temperature=temperature,
        repetition_penalty=1.1,
        max_length=4096,
        speaker=speaker,
    )

    output_as_np = output.audio.cpu().detach().numpy().squeeze()
    return output_as_np


def _speech_generation_parler(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    speaker_description: str,
    device: str,
) -> np.ndarray:
    input_ids = tokenizer(speaker_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()

    return waveform


def text_to_speech(
    input_text: str,
    model: Union[PreTrainedModel, InterfaceGGUFClass],
    tokenizer: PreTrainedTokenizerBase = None,
    speaker_profile: str = "",
    device: str = "cpu",
) -> np.ndarray:
    """
    Generates a speech waveform using the input_text, a model and a speaker profile to define a distinct voice pattern.

    Examples:
        >>> waveform = text_to_speech(input_text="Welcome to our amazing podcast", model=model, tokenizer=tokenizer, speaker_profile="Laura's voice is exciting and fast in delivery with very clear audio and no background noise.")

    Args:
        input_text (str): The text to convert to speech.
        model (PreTrainedModel): The model used for generating the waveform.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenizing the text in order to send to the model.
        speaker_profile (str): A description used by the ParlerTTS model to configure the speaker profile.
        device (str): The device to compute the generation on, such as "cuda:0" or "cpu".
    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    if isinstance(model, InterfaceGGUFClass):
        return _speech_generation_oute(input_text, model, speaker_profile)
    elif isinstance(model, PreTrainedModel):
        return _speech_generation_parler(
            input_text, model, tokenizer, speaker_profile, device
        )
    else:
        raise NotImplementedError("Model not yet implemented for TTS")
