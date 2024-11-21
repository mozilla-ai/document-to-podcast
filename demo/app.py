from pathlib import Path

import streamlit as st
from llama_cpp import Llama

from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS

PROMPT = """
Convert this text into a podcast script.
The conversation should be between 2 speakers.
Use [SPEAKER1] and [SPEAKER2] to limit sections. 
Do not include [INTRO], [OUTRO] or any other [SECTION].
Text:
"""

uploaded_file = st.file_uploader(
    "Choose a file", type=["pdf", "html", "txt", "docx", "md"]
)

if uploaded_file is not None:
    extension = Path(uploaded_file.name).suffix

    col1, col2 = st.columns(2)

    raw_text = DATA_LOADERS[extension](uploaded_file)
    with col1:
        st.title("Raw Text")
        st.text_area(
            f"Total Lenght: {len(raw_text)}",
            f"{raw_text[:500]} . . ."
        )

    clean_text = DATA_CLEANERS[extension](raw_text)
    with col2:
        st.title("Cleaned Text")
        st.text_area(
            f"Total Lenght: {len(clean_text)}",
            f"{clean_text[:500]} . . ."
        )

    # I set this value as a quick safeguard but we should actually tokenize the text and count the number of real tokens.
    if len(clean_text) > 4096 * 3:
        st.warning(f"Input text is too big ({len(clean_text)}). Using only a subset of it ({4096 * 3}).")
        clean_text = clean_text[:4096 * 3]

    with st.spinner('Downloading and Loading Model...'):
        llm = Llama.from_pretrained(
            repo_id="allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
            filename="olmoe-1b-7b-0924-instruct-q8_0.gguf",
            # 0 means that the model limit will be used, instead of the default (512) or other hardcoded value
            n_ctx=0
        )

    podcast = []
    with st.spinner('Writing Podcast Script...'):
        response = llm.create_chat_completion(
            messages = [
                {
                    "role": "user",
                    "content": f"{PROMPT}: {clean_text}"
                }
            ],
            stream=True
        )
        text = ""
        for item in response:
            if item["choices"][0].get("delta", {}).get("content", None):
                text += item["choices"][0].get("delta", {}).get("content", None)
            if text.endswith("\n"):
                st.write(text)
                text = ""
