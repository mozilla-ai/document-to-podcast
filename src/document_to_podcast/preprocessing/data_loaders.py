import PyPDF2
import requests

from docx import Document
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile


def load_pdf(pdf_file: str | UploadedFile) -> str | None:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.exception(e)
        return None


def load_txt(txt_file: str | UploadedFile) -> str | None:
    try:
        if isinstance(txt_file, UploadedFile):
            return txt_file.getvalue().decode("utf-8")
        else:
            with open(txt_file, "r") as file:
                return file.read()
    except Exception as e:
        logger.exception(e)
        return None


def load_docx(docx_file: str | UploadedFile) -> str | None:
    try:
        docx_reader = Document(docx_file)
        return "\n".join(paragraph.text for paragraph in docx_reader.paragraphs)
    except Exception as e:
        logger.exception(e)
        return None


def load_url(url: str) -> str | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.exception(e)
        return None
