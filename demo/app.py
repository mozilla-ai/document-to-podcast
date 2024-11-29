from pathlib import Path

import streamlit as st
from huggingface_hub import list_repo_files

from opennotebookllm.podcast_maker.config import PodcastConfig, SpeakerConfig
from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS
from opennotebookllm.inference.model_loaders import (
    load_llama_cpp_model,
    load_parler_tts_model_and_tokenizer,
)
from opennotebookllm.inference.text_to_speech import text_to_speech
from opennotebookllm.inference.text_to_text import text_to_text_stream


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

SPEAKER_DESCRIPTIONS = {
    "1": "Laura's voice is exciting and fast in delivery with very clear audio and no background noise.",
    "2": "Jon's voice is calm with very clear audio and no background noise."
}

st.title("Document To Podcast")

st.header("Uploading Data")

uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "docx", "md"]
)

if uploaded_file is not None:
    st.header("Loading and Cleaning")

    extension = Path(uploaded_file.name).suffix

    col1, col2 = st.columns(2)

    raw_text = DATA_LOADERS[extension](uploaded_file)
    with col1:
        st.subheader("Raw Text")
        st.text_area(f"Total Length: {len(raw_text)}", f"{raw_text[:500]} . . .")

    clean_text = DATA_CLEANERS[extension](raw_text)
    with col2:
        st.subheader("Cleaned Text")
        st.text_area(f"Total Length: {len(clean_text)}", f"{clean_text[:500]} . . .")

    with st.spinner("Downloading and Loading text-to-text Model..."):
        model = load_llama_cpp_model(model_id=f"{repo_name}/{model_name}")

    with st.spinner("Downloading and Loading text-to-speech Model..."):
        tts_model, tts_tokenizer = load_parler_tts_model_and_tokenizer(
            "parler-tts/parler-tts-mini-v1", "cpu"
        )

        # ~4 characters per token is considered a reasonable default.
        max_characters = model.n_ctx() * 4
        if len(clean_text) > max_characters:
            st.warning(
                f"Input text is too big ({len(clean_text)})."
                f" Using only a subset of it ({max_characters})."
            )
            clean_text = clean_text[:max_characters]

        system_prompt = st.text_area("Podcast generation prompt", value=PODCAST_PROMPT)

        if st.button("Generate Podcast"):
            with st.spinner("Generating Podcast..."):
                text = ""
                for chunk in text_to_text_stream(
                    clean_text, model, system_prompt=system_prompt.strip()
                ):
                    text += chunk
                    if text.endswith("\n") and "Speaker" in text:
                        st.write(text)
                        speaker = 
                        with st.spinner("Generating Audio..."):
                            waveform = parse_script_to_waveform(
                                text, demo_podcast_config
                            )
                        st.audio(waveform, sample_rate=demo_podcast_config.sampling_rate)
                        text = ""
