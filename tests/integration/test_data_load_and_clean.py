from document_to_podcast.preprocessing.data_cleaners import (
    clean_markdown,
    markdown_to_text,
)
from document_to_podcast.preprocessing.data_loaders import load_txt, load_file


def test_load_and_clean_pdf(example_data):
    file_path = example_data / "Mozilla-Trustworthy_AI.pdf"

    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = file_path.name
    mock_file.getvalue.return_value = open(file_path, "rb").read()

    result = load_file(mock_file)

    text = markdown_to_text(result)
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"


def test_load_and_clean_html(example_data):
    file_path = example_data / "introducing-mozilla-ai-investing-in-trustworthy-ai.html"

    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = file_path.name
    mock_file.getvalue.return_value = open(file_path, "rb").read()

    result = load_file(mock_file)

    text = markdown_to_text(result)
    assert text[:50] == "Skip to content Mozilla Internet Culture Deep Dive"


def test_load_and_clean_markdown(example_data):
    text = clean_markdown(load_txt(example_data / "Mozilla-Trustworthy_AI.md"))
    assert text[:50] == "Creating Trustworthy AI a Mozilla white paper on c"
