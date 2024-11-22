from pathlib import Path

import streamlit as st
from huggingface_hub import list_repo_files

from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS
from opennotebookllm.inference import load_LLama_model
from opennotebookllm.inference import text_to_text

PODCAST_PROMPT = """
Convert this text into a podcast script.
The conversation should be between 2 speakers.
Use [SPEAKER1] and [SPEAKER2] to limit sections.
Do not include [INTRO], [OUTRO] or any other [SECTION].
Text:
"""

REPO = "allenai/OLMoE-1B-7B-0924-Instruct-GGUF"

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

    # I set this value as a quick safeguard but we should actually tokenize the text and count the number of real tokens.
    if len(clean_text) > 4096 * 3:
        st.warning(
            f"Input text is too big ({len(clean_text)}). Using only a subset of it ({4096 * 3})."
        )
        clean_text = clean_text[: 4096 * 3]

    model_name = st.selectbox(
        "Select Model",
        [
            x
            for x in list_repo_files(REPO)
            if ".gguf" in x
            # The float16 is too big for the 16GB RAM codespace
            and "f16" not in x
        ],
        index=None,
    )
    if model_name:
        with st.spinner("Downloading and Loading Model..."):
            model = load_LLama_model(model_id=f"{REPO}/{model_name}")

        system_prompt = st.text_area("Podcast generation prompt", value=PODCAST_PROMPT)

        if st.button("Generate Podcast Script"):
            with st.spinner("Generating Podcast Script..."):
                text = ""
                for chunk in text_to_text(
                    clean_text, model, system_prompt=system_prompt.strip(), stream=True
                ):
                    text += chunk
                    if text.endswith("\n"):
                        st.write(text)
                        text = ""
