from document_to_podcast.preprocessing.data_cleaners import (
    clean_html,
    clean_with_regex,
    clean_markdown,
)
from document_to_podcast.preprocessing.data_loaders import load_pdf, load_txt


def test_load_and_clean_pdf(example_data):
    text = clean_with_regex(load_pdf(example_data / "Mozilla-Trustworthy_AI.pdf"))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"


def test_load_and_clean_html(example_data):
    text = clean_html(
        load_txt(
            example_data / "introducing-mozilla-ai-investing-in-trustworthy-ai.html"
        )
    )
    assert text[:50] == "Skip to content Mozilla Internet Culture Deep Dive"


def test_load_and_clean_markdown(example_data):
    text = clean_markdown(load_txt(example_data / "Mozilla-Trustworthy_AI.md"))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"
