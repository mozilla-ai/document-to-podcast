from typing import Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from pydantic import BaseModel, ConfigDict


class SpeakerConfig(BaseModel):
    """
    Pydantic model that stores configuration of an individual speaker for the TTS model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: PreTrainedModel
    speaker_id: str
    # ParlerTTS specific configuration
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    speaker_description: Optional[str] = (
        None  # This description is used by the ParlerTTS model to configure the speaker profile
    )


class PodcastConfig(BaseModel):
    """
    Pydantic model that stores configuration of all the speakers for the TTS model. This allows different speakers to
    use different models and configurations.
    """

    speakers: Dict[str, SpeakerConfig]
    sampling_rate: int = 44_100
