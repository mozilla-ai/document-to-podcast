import re
from pathlib import Path

import streamlit as st
from llama_cpp import Llama

from document_to_podcast.podcast_maker.config import SpeakerConfig, PodcastConfig
from document_to_podcast.preprocessing import DATA_LOADERS, DATA_CLEANERS
from document_to_podcast.inference.model_loaders import (
    load_llama_cpp_model,
    load_outetts_model,
    load_parler_tts_model_and_tokenizer,
)
from document_to_podcast.inference.text_to_speech import text_to_speech
from document_to_podcast.inference.text_to_text import text_to_text_stream


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
  "Speaker 1": "Welcome to our podcast! Today, we're exploring...",
  "Speaker 2": "Hi Laura! I'm excited to hear about this. Can you explain...",
  "Speaker 1": "Sure! Imagine it like this...",
  "Speaker 2": "Oh, that's cool! But how does..."
}
"""

# For a list of speakers supported: https://github.com/edwko/OuteTTS/tree/main/outetts/version/v1/default_speakers
SPEAKER_DESCRIPTIONS_OUTE = {
    "1": "female_1",
    "2": "male_1",
}
# For a list of speakers supported: https://github.com/huggingface/parler-tts?tab=readme-ov-file#-using-a-specific-speaker
SPEAKER_DESCRIPTIONS_PARLER = {
    "1": "Laura's voice is exciting and fast in delivery with very clear audio and no background noise.",
    "2": "Jon's voice is calm with very clear audio and no background noise.",
}

TTS_MODELS = [
    "OuteTTS-0.1-350M",
    "OuteTTS-0.2-500M",
    "parler-tts-large-v1",
    "parler-tts-mini-v1",
    "parler-tts-mini-expresso",
]


@st.cache_resource
def load_text_to_text_model() -> Llama:
    return load_llama_cpp_model(
        model_id="allenai/OLMoE-1B-7B-0924-Instruct-GGUF/olmoe-1b-7b-0924-instruct-q8_0.gguf"
    )


@st.cache_resource
def load_text_to_speech_model(model_id: str) -> PodcastConfig:
    if "oute" in model_id.lower():
        model = load_outetts_model(f"OuteAI/{model_id}-GGUF/{model_id}-FP16.gguf")
        tokenizer = None
        speaker_descriptions = SPEAKER_DESCRIPTIONS_OUTE
        sampling_rate = model.audio_codec.sr
    else:
        model, tokenizer = load_parler_tts_model_and_tokenizer(
            f"parler-tts/{model_id}", "cpu"
        )
        speaker_descriptions = SPEAKER_DESCRIPTIONS_PARLER
        sampling_rate = model.config.sampling_rate

    speaker_1 = SpeakerConfig(
        model=model,
        speaker_id="1",
        tokenizer=tokenizer,
        speaker_profile=speaker_descriptions["1"],
    )
    speaker_2 = SpeakerConfig(
        model=model,
        speaker_id="2",
        tokenizer=tokenizer,
        speaker_profile=speaker_descriptions["2"],
    )

    return PodcastConfig(
        speakers={s.speaker_id: s for s in [speaker_1, speaker_2]},
        sampling_rate=sampling_rate,
    )


st.title("Document To Podcast")

st.header("Uploading Data")

uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "docx", "md"]
)


if uploaded_file is not None:
    st.divider()
    st.header("Loading and Cleaning Data")
    st.markdown(
        "[Docs for this Step](https://mozilla-ai.github.io/document-to-podcast/step-by-step-guide/#step-1-document-pre-processing)"
    )
    st.divider()

    extension = Path(uploaded_file.name).suffix

    col1, col2 = st.columns(2)

    raw_text = DATA_LOADERS[extension](uploaded_file)
    with col1:
        st.subheader("Raw Text")
        st.text_area(
            f"Number of characters before cleaning: {len(raw_text)}",
            f"{raw_text[:500]} . . .",
        )

    clean_text = DATA_CLEANERS[extension](raw_text)
    with col2:
        st.subheader("Cleaned Text")
        st.text_area(
            f"Number of characters after cleaning: {len(clean_text)}",
            f"{clean_text[:500]} . . .",
        )

    st.divider()
    text_model = load_text_to_text_model()

    model_name = st.selectbox(
        label="Select Text-to-Speech Model", options=TTS_MODELS, index=None
    )

    if model_name:
        st.header("Downloading and Loading models")
        st.markdown(
            "[Docs for this Step](https://mozilla-ai.github.io/document-to-podcast/step-by-step-guide/#step-2-podcast-script-generation)"
        )
        st.divider()

        st.markdown(
            "For this demo, we are using [OLMoE-1B-7B-0924-Instruct-GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF) for text-to-text.\n"
        )
        st.markdown(
            "You can check the [Customization Guide](https://mozilla-ai.github.io/document-to-podcast/customization/)"
            " for more information on how to use different models."
        )
        tts_model = load_text_to_speech_model(model_name)

        # ~4 characters per token is considered a reasonable default.
        max_characters = text_model.n_ctx() * 4
        if len(clean_text) > max_characters:
            st.warning(
                f"Input text is too big ({len(clean_text)})."
                f" Using only a subset of it ({max_characters})."
            )
            clean_text = clean_text[:max_characters]

        st.divider()
        st.header("Podcast generation")
        st.markdown(
            "[Docs for this Step](https://mozilla-ai.github.io/document-to-podcast/step-by-step-guide/#step-3-audio-podcast-generation)"
        )
        st.divider()

        system_prompt = st.text_area("Podcast generation prompt", value=PODCAST_PROMPT)

        if st.button("Generate Podcast"):
            with st.spinner("Generating Podcast..."):
                text = ""
                for chunk in text_to_text_stream(
                    clean_text, text_model, system_prompt=system_prompt.strip()
                ):
                    text += chunk
                    if text.endswith("\n") and "Speaker" in text:
                        st.write(text)
                        speaker_id = re.search(r"Speaker (\d+)", text).group(1)
                        with st.spinner("Generating Audio..."):
                            speech = text_to_speech(
                                input_text=text.split(f'"Speaker {speaker_id}":')[-1],
                                model=tts_model.speakers[speaker_id].model,
                                tokenizer=tts_model.speakers[speaker_id].tokenizer,
                                speaker_profile=tts_model.speakers[
                                    speaker_id
                                ].speaker_profile,
                            )
                        st.audio(speech, sample_rate=tts_model.sampling_rate)
                        text = ""
