import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def text_to_speech(
    input_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tone: str,
) -> np.ndarray:
    """
    Generates a speech waveform from a text input using a pre-trained text-to-speech (TTS) model.

    Args:
        input_text (str): The text to convert to speech.
        model (PreTrainedModel): The model used for generating the waveform.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenizing the text in order to send to the model.
        tone (str): A description used by the ParlerTTS model to configure the speaker profile.
    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    input_ids = tokenizer(tone, return_tensors="pt").input_ids
    prompt_input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()
    return waveform
