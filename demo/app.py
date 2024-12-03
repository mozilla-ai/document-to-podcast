from pathlib import Path

import streamlit as st
from huggingface_hub import list_repo_files

from document_to_podcast.preprocessing import DATA_LOADERS, DATA_CLEANERS
from document_to_podcast.inference.model_loaders import load_llama_cpp_model
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
  "Speaker 1": "Welcome to our podcast! Today, we’re exploring...",
  "Speaker 2": "Hi Laura! I’m excited to hear about this. Can you explain...",
  "Speaker 1": "Sure! Imagine it like this...",
  "Speaker 2": "Oh, that’s cool! But how does..."
}
"""

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

        if st.button("Generate Podcast Script"):
            with st.spinner("Generating Podcast Script..."):
                text = ""
                for chunk in text_to_text_stream(
                    clean_text, model, system_prompt=system_prompt.strip()
                ):
                    text += chunk
                    if text.endswith("\n"):
                        st.write(text)
                        text = ""
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

        if st.button("Generate Podcast Script"):
            with st.spinner("Generating Podcast Script..."):
                text = ""
                for chunk in text_to_text_stream(
                    clean_text, model, system_prompt=system_prompt.strip()
                ):
                    text += chunk
                    if text.endswith("\n"):
                        st.write(text)
                        text = ""
