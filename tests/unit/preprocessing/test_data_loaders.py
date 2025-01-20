from unittest.mock import Mock

from document_to_podcast.preprocessing.data_loaders import (
    load_txt,
    load_url,
    load_file,
    load_url,
)


def test_load_pdf(example_data):
    file_path = example_data / "Mozilla-Trustworthy_AI.pdf"

    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = file_path.name
    mock_file.getvalue.return_value = open(file_path, "rb").read()

    result = load_file(mock_file)

    assert (
        "a Mozilla white paper on challenges and opportunities in the AI era" in result
    )


def test_load_invalid_pdf():
    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = "invalid.pdf"
    mock_file.getvalue.return_value = b""
    result = load_file(mock_file)

    assert result is None



def test_load_html(example_data):
    result = load_txt(
        example_data / "introducing-mozilla-ai-investing-in-trustworthy-ai.html"
    )
    assert (
        "A startup — and a community — building a trustworthy, independent, and open-source"
        in result
    )


def test_load_invalid_html():
    result = load_txt("invalid.html")
    assert result is None


def test_load_docx(example_data):
    file_path = example_data / "Mozilla-Trustworthy_AI.docx"

    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = file_path.name
    mock_file.getvalue.return_value = open(file_path, "rb").read()
    result = load_file(mock_file)

    assert (
        "a Mozilla white paper on challenges and opportunities in the AI era" in result
    )


def test_load_invalid_docx():
    # Create mock file using Mock
    mock_file = Mock()
    mock_file.name = "invalid.docx"
    mock_file.getvalue.return_value = b""
    result = load_file(mock_file)

    assert result is None


def test_load_markdown(example_data):
    result = load_txt(example_data / "Mozilla-Trustworthy_AI.md")
    assert (
        "a Mozilla white paper on challenges and opportunities in the AI era" in result
    )


def test_load_invalid_markdown():
    result = load_txt("invalid.md")
    assert result is None


def test_load_url():
    result = load_url(
        "https://blog.mozilla.ai/introducing-mozilla-ai-investing-in-trustworthy-ai/"
    )
    assert "Introducing Mozilla.ai: Investing in trustworthy AI" in result


def test_load_invalid_url():
    result = load_url("invalid")
    assert result is None
