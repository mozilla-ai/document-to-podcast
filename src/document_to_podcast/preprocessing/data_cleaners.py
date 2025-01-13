import re
from bs4 import BeautifulSoup
from markdown import markdown

def clean_with_regex(text: str) -> str:
    """
    Clean text using regular expressions.

    This function removes:
        - URLs
        - emails
        - special characters
        - extra spaces

    Examples:
        >>> clean_with_regex("\xa0Hello,   world! http://example.com")
        "Hello, world!"

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.[\w]+", "", text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:"\']', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_html(text: str) -> str:
    """Clean HTML text.

    This function removes:
        - scripts
        - styles
        - links
        - meta tags

    In addition, it calls [clean_with_regex][document_to_podcast.preprocessing.data_cleaners.clean_with_regex].

    Examples:
        >>> clean_html("<html><body><p>Hello,  world!  </p></body></html>"")
        "Hello, world!"

    Args:
        text (str): The HTML text to clean.

    Returns:
        str: The cleaned text.
    """
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "link", "meta"]):
        tag.decompose()
    text = soup.get_text()
    return clean_with_regex(text)


def clean_markdown(text: str) -> str:
    """Clean Markdown text.

    This function removes:
        - markdown images

    In addition, it calls [clean_with_regex][document_to_podcast.preprocessing.data_cleaners.clean_with_regex].

    Examples:
        >>> clean_markdown('# Title   with image ![alt text](image.jpg "Image Title")')
        "Title with image"

    Args:
        text (str): The Markdown text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'!\[.*?\]\(.*?(".*?")?\)', "", text)

    return clean_with_regex(text)


def markdown_to_text(markdown_string: str) -> str:
    """Converts a Markdown string to plain text.

    This function converts a Markdown string to HTML, removes code snippets,
    then extracts the text content, removes extra line breaks and whitespace.

    Source: https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe

    Args:
        markdown_string (str): The Markdown string to convert.

    Returns:
        str: The plain text representation of the Markdown string.

    Examples:
        >>> markdown_to_text("# Heading\\nSome text with `code` and <pre>preformatted</pre>")
        'Heading Some text with code and preformatted'
    """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_string, "html.parser")

    # Extract all text content
    text_content = "".join(soup.findAll(string=True))

    # Remove leading/trailing whitespace and replace multiple newlines with a single space
    text_content = text_content.strip()
    text_content = re.sub(r"\n+", " ", text_content)

    return text_content
