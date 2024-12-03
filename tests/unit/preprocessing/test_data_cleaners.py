from document_to_podcast.preprocessing.data_cleaners import (
    clean_html,
    clean_with_regex,
    clean_markdown,
)


def test_clean_html():
    text = "<html><body><p>Hello, world!</p></body></html>"
    cleaned_text = clean_html(text)
    assert cleaned_text == "Hello, world!"


def test_clean_with_regex():
    text = "\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\n\xa0\nThis work is licensed under the Creative Commons Attribution 4.0 (BY) license, which means\n\xa0\nthat the text may be remixed, transformed and built upon, and be copied and redistributed in\n\xa0\nany medium or format even commercially, provided credit is given to the author. For details go\n\xa0\nto http://creativecommons.org/licenses/by/4.0/\n\xa0\n\xa0\n\xa0\n"
    cleaned_text = clean_with_regex(text)
    assert (
        cleaned_text
        == "This work is licensed under the Creative Commons Attribution 4.0 BY license, which means that the text may be remixed, transformed and built upon, and be copied and redistributed in any medium or format even commercially, provided credit is given to the author. For details go to"
    )


def test_clean_markdown():
    text = """
    # This is some text with an image
    ![alt text](image.jpg "Image Title") and another one ![alt text2](another_image.png).
    - Item list here.
    """
    cleaned_text = clean_markdown(text)
    assert (
        cleaned_text
        == "This is some text with an image and another one . Item list here."
    )
