import streamlit as st

from opennotebookllm.preprocessing.data_loaders import load_pdf
from opennotebookllm.preprocessing.data_cleaners import clean_html, clean_pdf


uploaded_file = st.file_uploader("Choose a file", type=["pdf", "html"])

if uploaded_file is not None:
    if uploaded_file.type == "text/html":
        raw_text = uploaded_file.getvalue().decode("utf-8")
        clean_text = clean_html(raw_text)
    elif uploaded_file.type == "application/pdf":
        raw_text = load_pdf(uploaded_file)
        clean_text = clean_pdf(raw_text)
    col1, col2 = st.columns(2)
    with col1:
        st.title("Raw Text")
        st.write(raw_text[:200])
    with col2:
        st.title("Cleaned Text")
        st.write(clean_text[:200])
