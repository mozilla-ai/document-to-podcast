from pathlib import Path

import streamlit as st

from opennotebookllm.preprocessing import DATA_LOADERS, DATA_CLEANERS
from opennotebookllm.text_to_podcast import load_model
from opennotebookllm.text_to_podcast import text_to_podcast


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

    with st.spinner("Downloading and Loading Model..."):
        model = load_model()

    with st.spinner("Writing Podcast Script..."):
        text = ""
        for chunk in text_to_podcast(clean_text, model, stream=True):
            text += chunk
            if text.endswith("\n"):
                st.write(text)
                text = ""
