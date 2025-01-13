from document_to_podcast.preprocessing.data_cleaners import (
    clean_markdown,
    markdown_to_text
)
from document_to_podcast.preprocessing.data_loaders import load_txt, load_file


def test_load_and_clean_pdf(example_data):
    text = markdown_to_text(
        load_file(str(example_data / "Mozilla-Trustworthy_AI.pdf")))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"


def test_load_and_clean_html(example_data):
    text = markdown_to_text(load_file(
        str(example_data / "introducing-mozilla-ai-investing-in-trustworthy-ai.html")))
    assert text[:50] == "Skip to content Mozilla Internet Culture Deep Dive"


def test_load_and_clean_markdown(example_data):
    text = clean_markdown(load_txt(example_data / "Mozilla-Trustworthy_AI.md"))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"
