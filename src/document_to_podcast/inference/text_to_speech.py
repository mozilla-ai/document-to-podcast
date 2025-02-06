import numpy as np
from kokoro import KPipeline

from document_to_podcast.inference.model_loaders import TTSModel


def _text_to_speech_kokoro(
    input_text: str, model: KPipeline, voice_profile: str
) -> np.ndarray:
    """
    TTS generation function for the Kokoro model
    Args:
        input_text (str): The text to convert to speech.
        model (KPipeline): The kokoro pipeline as defined in https://github.com/hexgrad/kokoro
        voice_profile (str) : a pre-defined ID for the Kokoro models (e.g. "af_bella")
            more info here https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

    Returns:
        numpy array: The waveform of the speech as a 2D numpy array
    """
    generator = model(input_text, voice=voice_profile)

    _, _, audio = next(generator)  # returns graphemes/text, phonemes, audio

    return np.array(audio)


TTS_INFERENCE = {
    # To add support for your model, add it here in the format {model_id} : _inference_function
    "hexgrad/Kokoro-82M": _text_to_speech_kokoro,
}


def text_to_speech(input_text: str, model: TTSModel, voice_profile: str) -> np.ndarray:
    """
    Generate speech from text using a TTS model.

    Args:
        input_text (str): The text to convert to speech.
        model (TTSModel): The TTS model to use.
        voice_profile (str): The voice profile to use for the speech. The format depends on the TTSModel used.
    Returns:
        np.ndarray: The waveform of the speech as a 2D numpy array
    """
    return TTS_INFERENCE[model.model_id](
        input_text, model.model, voice_profile, **model.custom_args
    )
