from .data_loaders import load_pdf, load_txt, load_docx, load_url
from .data_cleaners import clean_with_regex, clean_html, clean_markdown


DATA_LOADERS = {
    ".docx": load_docx,
    ".html": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".txt": load_txt,
    "url": load_url,
}

DATA_CLEANERS = {
    ".docx": clean_with_regex,
    ".html": clean_html,
    ".md": clean_markdown,
    ".pdf": clean_with_regex,
    ".txt": clean_with_regex,
}
