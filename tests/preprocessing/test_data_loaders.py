from opennotebookllm.preprocessing.data_loaders import load_pdf, load_txt


def test_load_pdf(example_data):
    result = load_pdf(example_data / "Mozilla-Trustworthy_AI.pdf")
    assert (
        "a Mozilla white paper on challenges and opportunities in the AI era" in result
    )


def test_load_invalid_pdf():
    result = load_pdf("invalid.pdf")
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
