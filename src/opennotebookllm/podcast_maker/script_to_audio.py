import wave

import numpy as np

from opennotebookllm.inference.model_loaders import load_parler_tts_model_and_tokenizer
from opennotebookllm.inference.text_to_speech import text_to_speech

from opennotebookllm.podcast_maker.config import PodcastConfig, SpeakerConfig


def parse_script_to_waveform(script: str, podcast_config: PodcastConfig):
    """
    Given a script with speaker identifiers (such as "Speaker 1") parse it so that each speaker has its own unique
    voice and concatenate all the voices in a sequence to form the complete podcast.
    Args:
        script:
        podcast_config:

    Returns: A 2D numpy array containing the whole podcast in waveform format.

    """
    parts = script.split("Speaker ")
    podcast_waveform = []
    for part in parts:
        if ":" in part:
            speaker_id, speaker_text = part.split(":")
            speaker_model = podcast_config.speakers[speaker_id].model
            speaker_tokenizer = podcast_config.speakers[speaker_id].tokenizer
            speaker_description = podcast_config.speakers[
                speaker_id
            ].speaker_description

            speaker_waveform = text_to_speech(
                speaker_text, speaker_model, speaker_tokenizer, speaker_description
            )
            podcast_waveform.append(speaker_waveform)

    return np.concatenate(podcast_waveform)


def save_waveform_as_file(
    waveform: np.ndarray, sampling_rate: int, filename: str
) -> None:
    # Normalize the float32 waveform into int16, which is what the wave module requires
    waveform_int16 = np.int16(
        waveform / np.max(np.abs(waveform)) * 32767
    )  # 32767 = max int16
    with wave.open(filename, "w") as f:
        f.setnchannels(1)  # 2 for Stereo, 1 for Mono
        f.setsampwidth(2)  # bytes per sample
        f.setframerate(sampling_rate)
        f.writeframes(waveform_int16.tobytes())


if __name__ == "__main__":
    test_filename = "test_podcast.wav"
    test_podcast_script = (
        "Speaker 1: Welcome to our podcast. "
        "Speaker 2: It's great to be here!"
        "Speaker 1: So what do you want to talk about today? "
        "Speaker 2: I wish I had a clue!"
    )

    model, tokenizer = load_parler_tts_model_and_tokenizer(
        "parler-tts/parler-tts-mini-v1", "cpu"
    )
    speaker_1 = SpeakerConfig(
        model=model,
        speaker_id="1",
        tokenizer=tokenizer,
        speaker_description="Laura's voice is exciting and fast in delivery with very clear audio and no background noise.",
    )
    speaker_2 = SpeakerConfig(
        model=model,
        speaker_id="2",
        tokenizer=tokenizer,
        speaker_description="Jon's voice is calm with very clear audio and no background noise.",
    )
    demo_podcast_config = PodcastConfig(
        speakers={s.speaker_id: s for s in [speaker_1, speaker_2]}
    )
    test_podcast_waveform = parse_script_to_waveform(
        test_podcast_script, demo_podcast_config
    )

    save_waveform_as_file(
        test_podcast_waveform,
        sampling_rate=demo_podcast_config.sampling_rate,
        filename=test_filename,
    )
