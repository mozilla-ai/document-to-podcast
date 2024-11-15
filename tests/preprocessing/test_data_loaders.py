from opennotebookllm.preprocessing.data_loaders import load_pdf


def test_load_pdf(example_data):
    result = load_pdf(example_data / "Mozilla-Trustworthy_AI.pdf")
    assert (
        "a Mozilla white paper on challenges and opportunities in the AI era" in result
    )
