import logging
from pathlib import Path
import PyPDF2
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks using Langchain's RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise

def read_file(file_path: str) -> str:
    """Read content from a file (PDF or TXT)"""
    try:
        if file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise 