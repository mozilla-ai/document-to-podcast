import numpy as np
from outetts.version.v1.interface import InterfaceGGUF

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


TTS_INFERENCE = {
    # To add support for your model, add it here in the format {model_id} : _inference_function
    "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf": _text_to_speech_oute,
    "OuteAI/OuteTTS-0.2-500M-GGUF/OuteTTS-0.2-500M-FP16.gguf": _text_to_speech_oute,
}


def text_to_speech(input_text: str, model: TTSModel, voice_profile: str) -> np.ndarray:
    return TTS_INFERENCE[model.model_id](
        input_text, model.model, voice_profile, **model.custom_args
    )
