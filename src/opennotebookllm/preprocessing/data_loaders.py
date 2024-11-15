import PyPDF2
import PyPDF2.errors

from loguru import logger


def load_pdf(pdf_file):
    try:
        with open(pdf_file, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.exception(e)
        return None


def load_txt(txt_file):
    try:
        with open(txt_file, "r") as file:
            return file.read()
    except Exception as e:
        logger.exception(e)
        return None
