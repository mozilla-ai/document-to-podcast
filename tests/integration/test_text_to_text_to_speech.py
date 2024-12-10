import os
from pathlib import Path

from document_to_podcast.inference.model_loaders import load_llama_cpp_model
from document_to_podcast.inference.text_to_speech import text_to_speech
from document_to_podcast.inference.text_to_text import text_to_text
from document_to_podcast.podcast_maker.config import PodcastConfig
from document_to_podcast.podcast_maker.script_to_audio import save_waveform_as_file


def test_text_to_text_to_speech(tmp_path: Path, podcast_config: PodcastConfig):
    model = load_llama_cpp_model(
        "HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF/smollm-135m-instruct-add-basics-q8_0.gguf"
    )
    result = text_to_text(
        "What is the capital of France?",
        model=model,
        system_prompt="",
        return_json=False,
        stop=".",
    )

    speaker_cfg = list(podcast_config.speakers.values())[0]
    waveform = text_to_speech(
        input_text=result,
        model=speaker_cfg.model,
        tokenizer=speaker_cfg.tokenizer,
        speaker_profile=speaker_cfg.speaker_profile,
    )

    filename = str(tmp_path / "test_text_to_text_to_speech_parler.wav")
    save_waveform_as_file(
        waveform=waveform, sampling_rate=podcast_config.sampling_rate, filename=filename
    )

    assert os.path.isfile(filename)
