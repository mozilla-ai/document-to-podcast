from pathlib import Path
from typing import Literal
from typing_extensions import Annotated

from pydantic import BaseModel, FilePath
from pydantic.functional_validators import AfterValidator

from document_to_podcast.preprocessing import DATA_LOADERS


DEFAULT_PROMPT = """
You are a podcast scriptwriter generating engaging and natural-sounding conversations in JSON format.
The script features the following speakers:
{SPEAKERS}
Instructions:
- Write dynamic, easy-to-follow dialogue.
- Include natural interruptions and interjections.
- Avoid repetitive phrasing between speakers.
- Format output as a JSON conversation.
Example:
{
  "Speaker 1": "Welcome to our podcast! Today, we're exploring...",
  "Speaker 2": "Hi! I'm excited to hear about this. Can you explain...",
  "Speaker 1": "Sure! Imagine it like this...",
  "Speaker 2": "Oh, that's cool! But how does..."
}
"""

DEFAULT_SPEAKERS = [
    {
        "id": 1,
        "name": "Laura",
        "description": "The main host. She explains topics clearly using anecdotes and analogies, teaching in an engaging and captivating way.",
        "tone": "Laura's voice is exciting and fast in delivery with very clear audio and no background noise.",
    },
    {
        "id": 2,
        "name": "Jon",
        "description": "The co-host. He keeps the conversation on track, asks curious follow-up questions, and reacts with excitement or confusion, often using interjections like hmm or umm.",
        "tone": "Jon's voice is calm with very clear audio and no background noise.",
    },
]


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


def validate_text_to_text_prompt(value):
    if "{SPEAKERS}" not in value:
        raise ValueError("text_to_text_prompt must contain `{SPEAKERS}` placeholder")
    return value


class Speaker(BaseModel):
    id: int
    name: str
    description: str
    tone: str

    def __str__(self):
        return f"Speaker {self.id}. Named {self.name}. {self.description}"


class Config(BaseModel):
    input_file: Annotated[FilePath, AfterValidator(validate_input_file)]
    output_folder: str
    text_to_text_model: Annotated[str, AfterValidator(validate_text_to_text_model)]
    text_to_text_prompt: str
    text_to_speech_model: Literal[
        "parler-tts/parler-tts-large-v1",
        "parler-tts/parler-tts-mini-v1",
        "parler-tts/parler-tts-mini-v1.1",
    ]
    speakers: list[Speaker]
    sampling_rate: int