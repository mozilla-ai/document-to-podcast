from pathlib import Path
from typing_extensions import Annotated

import yaml
from fire import Fire
from loguru import logger
from pydantic import BaseModel, Field, FilePath
from pydantic.functional_validators import AfterValidator


from document_to_podcast.inference.model_loaders import load_llama_cpp_model
from document_to_podcast.inference.text_to_text import text_to_text_stream
from document_to_podcast.preprocessing import DATA_CLEANERS, DATA_LOADERS


PODCAST_PROMPT = """
You are a podcast scriptwriter generating engaging and natural-sounding conversations in JSON format. The script features two speakers:
Speaker 1: Laura, the main host. She explains topics clearly using anecdotes and analogies, teaching in an engaging and captivating way.
Speaker 2: Jon, the co-host. He keeps the conversation on track, asks curious follow-up questions, and reacts with excitement or confusion, often using interjections like “hmm” or “umm.”
Instructions:
- Write dynamic, easy-to-follow dialogue.
- Include natural interruptions and interjections.
- Avoid repetitive phrasing between speakers.
- Format output as a JSON conversation.
Example:
{
  "Speaker 1": "Welcome to our podcast! Today, we’re exploring...",
  "Speaker 2": "Hi Laura! I’m excited to hear about this. Can you explain...",
  "Speaker 1": "Sure! Imagine it like this...",
  "Speaker 2": "Oh, that’s cool! But how does..."
}
"""


def validate_input_file(value):
    if Path(value).suffix not in DATA_LOADERS:
        raise ValueError(
            f"input_file extension must be one of {list(DATA_LOADERS.keys())}"
        )
    return value


def validate_text_to_text_model(value):
    parts = value.split("/")
    if len(parts) != 3:
        raise ValueError("text_to_text_model must be formatted as `owner/repo/file`")
    if not value.endswith(".gguf"):
        raise ValueError("text_to_text_model must be a gguf file")
    return value


class Config(BaseModel):
    input_file: Annotated[FilePath, AfterValidator(validate_input_file)]
    output_folder: str
    text_to_text_model: Annotated[str, AfterValidator(validate_text_to_text_model)]
    text_to_text_prompt: str = Field(default=PODCAST_PROMPT)


@logger.catch(reraise=True)
def document_to_podcast(
    input_file: str | None = None,
    output_folder: str | None = None,
    text_to_text_model: str = "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf",
    text_to_text_prompt: str = PODCAST_PROMPT,
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
            Defaults to "allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf".
        text_to_text_prompt (str, optional): The prompt for the text-to-text model.
        from_config (str, optional): The path to the config file. Defaults to None.
    """
    if from_config:
        config = Config.model_validate(yaml.safe_load(Path(from_config).read_text()))
    else:
        config = Config(
            input_file=input_file,
            output_folder=output_folder,
            text_to_text_model=text_to_text_model,
            text_to_text_prompt=text_to_text_prompt,
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
    model = load_llama_cpp_model(model_id=config.text_to_text_model)

    # ~4 characters per token is considered a reasonable default.
    max_characters = model.n_ctx() * 4
    if len(clean_text) > max_characters:
        logger.warning(
            f"Input text is too big ({len(clean_text)})."
            f" Using only a subset of it ({max_characters})."
        )
    clean_text = clean_text[:max_characters]

    logger.info("Generating Podcast...")
    podcast_script = ""
    text = ""
    for chunk in text_to_text_stream(
        clean_text, model, system_prompt=config.text_to_text_prompt.strip()
    ):
        text += chunk
        podcast_script += chunk
        if text.endswith("\n"):
            logger.debug(text)
            text = ""
    (output_folder / "podcast.txt").write_text(podcast_script)


def main():
    Fire(document_to_podcast)
