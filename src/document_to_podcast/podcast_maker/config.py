from typing import Dict, Optional, Union

from parler_tts import ParlerTTSForConditionalGeneration
from outetts.version.v1.interface import InterfaceGGUF as InterfaceGGUFClass
from transformers import PreTrainedTokenizerBase
from pydantic import BaseModel, ConfigDict


class SpeakerConfig(BaseModel):
    """
    Pydantic model that stores configuration of an individual speaker for the TTS model.

     model: The actual model instance to be used for generation.
     speaker_id: A string defining the speaker in order to have a consistent voice during podcast generation.
     speaker_profile: This profile is defined based on the specific model family used (e.g. Parler uses natural language descriptions whereas Oute models use IDs)
     tokenizer: Parler models also need a tokenizer. This is None in the case of Oute models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Union[ParlerTTSForConditionalGeneration, InterfaceGGUFClass]
    speaker_id: str
    speaker_profile: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None


class PodcastConfig(BaseModel):
    """
    Pydantic model that stores configuration of all the speakers for the TTS model. This allows different speakers to
    use different models and configurations.
    """

    speakers: Dict[str, SpeakerConfig]
    sampling_rate: int = 44_100
