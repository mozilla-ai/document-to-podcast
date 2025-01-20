import os
import PyPDF2
from pathlib import Path
import requests
from tempfile import NamedTemporaryFile

from docx import Document
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile
from markitdown import MarkItDown


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


def load_file(file: str | UploadedFile) -> str | None:
    """
    Loads the content of a file and converts it to markdown.

    Args:
        file: Either a file path string or a Streamlit UploadedFile object

    Returns:
        The markdown text content if successful, None otherwise
    """
    if not hasattr(file, 'name'):
        logger.error("Invalid file object: missing 'name' attribute")
        return None

    tmp_file_path = None
    try:
        # Create temporary file
        extension = Path(file.name).suffix
        with NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            content = file.getvalue()
            if not content:
                logger.error("File content is empty")
                return None
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Convert to markdown
            markdown_converter = MarkItDown()
            markdown_content = markdown_converter.convert(tmp_file_path)
            return markdown_content.text_content
        except Exception as e:
            logger.error(f"Unexpected conversion error: {str(e)}")
            return None

    except Exception as e:
        logger.exception(f"Unexpected error while processing file: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if tmp_file_path and Path(tmp_file_path).exists():
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to remove temporary file {tmp_file_path}: {str(e)}")

def load_url(url: str) -> str | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.exception(e)
        return None
