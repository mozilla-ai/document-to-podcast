from io import BytesIO

import PyPDF2
import PyPDF2.errors
from loguru import logger


def load_pdf(pdf_file: str | BytesIO) -> str | None:
    try:
        if isinstance(pdf_file, str):
            with open(pdf_file, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
        else:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.exception(e)
        return None


def load_txt(txt_file: str) -> str | None:
    try:
        with open(txt_file, "r") as file:
            return file.read()
    except Exception as e:
        logger.exception(e)
        return None


LOADERS = {
    "pdf": load_pdf,
    "txt": load_txt,
    "html": load_txt,
}
