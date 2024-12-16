import re
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from fire import Fire
from loguru import logger

from document_to_podcast.config import (
    Config,
    Speaker,
    DEFAULT_PROMPT,
    DEFAULT_SPEAKERS,
    SUPPORTED_TTS_MODELS,
)
from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_outetts_model,
    load_parler_tts_model_and_tokenizer,
)
from document_to_podcast.inference.text_to_text import text_to_text_stream
from document_to_podcast.inference.text_to_speech import text_to_speech
from document_to_podcast.preprocessing import DATA_CLEANERS, DATA_LOADERS


@logger.catch(reraise=True)
def document_to_podcast(
    input_file: str | None = None,
    output_folder: str | None = None,
    text_to_text_model: str = "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf",
    text_to_text_prompt: str = DEFAULT_PROMPT,
    text_to_speech_model: SUPPORTED_TTS_MODELS = "OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf",
    speakers: list[Speaker] | None = None,
    from_config: str | None = None,
):
    """
    Generate a podcast from a document.

    Args:
        input_file (str): The path to the input file.
            Supported extensions:

                - .pdf
                - .html
                - .txt
                - .docx
                - .md

        output_folder (str): The path to the output folder.
            Two files will be created:

                - {output_folder}/podcast.txt
                - {output_folder}/podcast.wav

        text_to_text_model (str, optional): The path to the text-to-text model.

            Need to be formatted as `owner/repo/file`.

            Need to be a gguf file.

            Defaults to `allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf`.

        text_to_text_prompt (str, optional): The prompt for the text-to-text model.
            Defaults to DEFAULT_PROMPT.

        text_to_speech_model (str, optional): The path to the text-to-speech model.
            Defaults to `OuteAI/OuteTTS-0.1-350M-GGUF/OuteTTS-0.1-350M-FP16.gguf`.

        speakers (list[Speaker] | None, optional): The speakers for the podcast.
            Defaults to DEFAULT_SPEAKERS.

        from_config (str, optional): The path to the config file. Defaults to None.


            If provided, all other arguments will be ignored.
    """
    if from_config:
        config = Config.model_validate(yaml.safe_load(Path(from_config).read_text()))
    else:
        speakers = speakers or DEFAULT_SPEAKERS
        config = Config(
            input_file=input_file,
            output_folder=output_folder,
            text_to_text_model=text_to_text_model,
            text_to_text_prompt=text_to_text_prompt,
            text_to_speech_model=text_to_speech_model,
            speakers=[Speaker.model_validate(speaker) for speaker in speakers],
        )

    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    data_loader = DATA_LOADERS[Path(config.input_file).suffix]
    logger.info(f"Loading {config.input_file}")
    raw_text = data_loader(config.input_file)
    logger.debug(f"Loaded {len(raw_text)} characters")

    data_cleaner = DATA_CLEANERS[Path(config.input_file).suffix]
    logger.info(f"Cleaning {config.input_file}")
    clean_text = data_cleaner(raw_text)
    logger.debug(f"Cleaned {len(raw_text) - len(clean_text)} characters")
    logger.debug(f"Length of cleaned text: {len(clean_text)}")

    logger.info(f"Loading {config.text_to_text_model}")
    text_model = load_llama_cpp_model(model_id=config.text_to_text_model)

    logger.info(f"Loading {config.text_to_speech_model}")
    if "oute" in config.text_to_speech_model.lower():
        speech_model = load_outetts_model(model_id=config.text_to_speech_model)
        speech_tokenizer = None
        sample_rate = speech_model.audio_codec.sr
    else:
        speech_model, speech_tokenizer = load_parler_tts_model_and_tokenizer(
            model_id=config.text_to_speech_model
        )
        sample_rate = speech_model.config.sampling_rate

    # ~4 characters per token is considered a reasonable default.
    max_characters = text_model.n_ctx() * 4
    if len(clean_text) > max_characters:
        logger.warning(
            f"Input text is too big ({len(clean_text)})."
            f" Using only a subset of it ({max_characters})."
        )
    clean_text = clean_text[:max_characters]

    logger.info("Generating Podcast...")
    podcast_script = ""
    text = ""
    podcast_audio = []
    system_prompt = config.text_to_text_prompt.strip()
    system_prompt = system_prompt.replace(
        "{SPEAKERS}", "\n".join(str(speaker) for speaker in config.speakers)
    )
    for chunk in text_to_text_stream(
        clean_text, text_model, system_prompt=system_prompt
    ):
        text += chunk
        podcast_script += chunk
        if text.endswith("\n") and "Speaker" in text:
            logger.debug(text)
            speaker_id = re.search(r"Speaker (\d+)", text).group(1)
            voice_profile = next(
                speaker.voice_profile
                for speaker in config.speakers
                if speaker.id == int(speaker_id)
            )
            speech = text_to_speech(
                text.split(f'"Speaker {speaker_id}":')[-1],
                speech_model,
                voice_profile,
                tokenizer=speech_tokenizer,  # Applicable only for parler models
            )
            podcast_audio.append(speech)
            text = ""

    logger.info("Saving Podcast...")
    sf.write(
        str(output_folder / "podcast.wav"),
        np.concatenate(podcast_audio),
        samplerate=sample_rate,
    )
    (output_folder / "podcast.txt").write_text(podcast_script)
    logger.success("Done!")


def main():
    Fire(document_to_podcast)
