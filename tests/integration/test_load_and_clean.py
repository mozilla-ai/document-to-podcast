from opennotebookllm.preprocessing.data_cleaners import clean_html, clean_pdf
from opennotebookllm.preprocessing.data_loaders import load_pdf, load_txt


def test_load_and_clean_pdf(example_data):
    text = clean_pdf(load_pdf(example_data / "Mozilla-Trustworthy_AI.pdf"))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"


def test_load_and_clean_html(example_data):
    text = clean_html(
        load_txt(
            example_data / "introducing-mozilla-ai-investing-in-trustworthy-ai.html"
        )
    )
    assert text[:50] == "Skip to content Mozilla Internet Culture Deep Dive"
