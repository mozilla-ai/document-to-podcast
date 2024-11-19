import re
from bs4 import BeautifulSoup


def clean_with_regex(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.[\w]+", "", text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:"\']', "", text)
    return text


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "link", "meta"]):
        tag.decompose()
    text = soup.get_text()
    return clean_with_regex(text)


CLEANERS = {"pdf": clean_with_regex, "html": clean_html, "txt": clean_with_regex}
