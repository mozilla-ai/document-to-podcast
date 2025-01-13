from .data_loaders import load_pdf, load_txt, load_docx, load_file
from .data_cleaners import clean_with_regex, clean_html, clean_markdown, markdown_to_text


DATA_LOADERS = {
    ".docx": load_file,
    ".html": load_txt,
    ".md": load_txt,
    ".pdf": load_file,
    ".txt": load_txt,
    "url": load_url,
}

DATA_CLEANERS = {
    ".docx": markdown_to_text,
    ".html": clean_html,
    ".md": clean_markdown,
    ".pdf": markdown_to_text,
    ".txt": clean_with_regex,
}
