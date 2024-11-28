from pathlib import Path

import streamlit as st
from huggingface_hub import list_repo_files

from opennotebookllm.podcast_maker.config import PodcastConfig, SpeakerConfig
from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS
from opennotebookllm.inference.model_loaders import (
    load_llama_cpp_model,
    load_parler_tts_model_and_tokenizer,
)
from opennotebookllm.inference.text_to_text import text_to_text_stream
from opennotebookllm.podcast_maker.script_to_audio import (
    parse_script_to_waveform,
    save_waveform_as_file,
)

PODCAST_PROMPT = """
You are a helpful podcast writer.
You will take the input text and generate a conversation between 2 speakers.
Example of response:
{
    "Speaker 1": "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're going to dive into the exciting world of TrustWorthy AI.",
    "Speaker 2": "Hi, I'm excited to be here, so what is TrustWorthy AI?",
    "Speaker 1":"Ah, great question! It is a term used by the European High Level Expert Group on AI. Mozilla defines trustworthy AI as AI that is demonstrably worthy of trust, tech that considers accountability, agency, and individual and collective well-being."
}
"""

SPEAKER_1_DESC = "Laura's voice is exciting and fast in delivery with very clear audio and no background noise."
SPEAKER_2_DESC = "Jon's voice is calm with very clear audio and no background noise."

CURATED_REPOS = [
    "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
    "MaziyarPanahi/SmolLM2-1.7B-Instruct-GGUF",
    # system prompt seems to be ignored for this model.
    # "microsoft/Phi-3-mini-4k-instruct-gguf",
    "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-3B-Instruct-GGUF",
]

uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "docx", "md"]
)

if uploaded_file is not None:
    extension = Path(uploaded_file.name).suffix

    col1, col2 = st.columns(2)

    raw_text = DATA_LOADERS[extension](uploaded_file)
    with col1:
        st.title("Raw Text")
        st.text_area(f"Total Length: {len(raw_text)}", f"{raw_text[:500]} . . .")

    clean_text = DATA_CLEANERS[extension](raw_text)
    with col2:
        st.title("Cleaned Text")
        st.text_area(f"Total Length: {len(clean_text)}", f"{clean_text[:500]} . . .")

    repo_name = st.selectbox("Select Repo", CURATED_REPOS)
    model_name = st.selectbox(
        "Select Model",
        [
            x
            for x in list_repo_files(repo_name)
            if ".gguf" in x.lower() and ("q8" in x.lower() or "fp16" in x.lower())
        ],
        index=None,
    )
    if model_name:
        with st.spinner("Downloading and Loading Model..."):
            model = load_llama_cpp_model(model_id=f"{repo_name}/{model_name}")

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
            final_script = ""
            with st.spinner("Generating Podcast Script..."):
                text = ""
                for chunk in text_to_text_stream(
                    clean_text, model, system_prompt=system_prompt.strip()
                ):
                    text += chunk
                    final_script += chunk
                    if text.endswith("\n"):
                        st.write(text)
                        text = ""

            if final_script:
                model.close()  # Free up memory in order to load the TTS model

                filename = "demo_podcast.wav"

                with st.spinner("Downloading and Loading TTS Model..."):
                    tts_model, tokenizer = load_parler_tts_model_and_tokenizer(
                        "parler-tts/parler-tts-mini-v1", "cpu"
                    )
                speaker_1 = SpeakerConfig(
                    model=tts_model,
                    speaker_id="1",
                    tokenizer=tokenizer,
                    speaker_description=SPEAKER_1_DESC,
                )
                speaker_2 = SpeakerConfig(
                    model=tts_model,
                    speaker_id="2",
                    tokenizer=tokenizer,
                    speaker_description=SPEAKER_2_DESC,
                )
                demo_podcast_config = PodcastConfig(
                    speakers={s.speaker_id: s for s in [speaker_1, speaker_2]}
                )

                with st.spinner("Generating Audio..."):
                    waveform = parse_script_to_waveform(
                        final_script, demo_podcast_config
                    )
                save_waveform_as_file(
                    waveform, demo_podcast_config.sampling_rate, filename
                )
                st.audio(filename)
