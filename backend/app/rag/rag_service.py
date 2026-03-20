# backend/app/rag/rag_service.py

from app.rag.rag_chain import run_rag_chain, stream_rag_chain
from app.rag.ingest import ingest_document
from app.logger.logger import logger
from typing import Iterator


def get_rag_response(query: str) -> dict:
    """
    Public entry point — standard (non-streaming) RAG response.

    Args:
        query : user's question

    Returns:
        dict:
            - answer  : LLM answer string
            - sources : list of source filenames used
    """
    logger.info(f"[rag_service] Query: '{query}'")
    return run_rag_chain(query)


def stream_rag_response(query: str) -> Iterator[str]:
    """
    Public entry point — streaming RAG response.
    Yields token chunks as they arrive from Groq.

    Args:
        query : user's question

    Yields:
        str token chunks
    """
    logger.info(f"[rag_service] Streaming query: '{query}'")
    yield from stream_rag_chain(query)


def ingest_file(file_path: str) -> dict:
    """
    Public entry point — ingest a document into the FAISS vectorstore.

    Args:
        file_path : path to the uploaded document (PDF / TXT / DOCX)

    Returns:
        dict:
            - file     : filename
            - chunks   : number of chunks ingested
            - message  : success message
    """
    logger.info(f"[rag_service] Ingesting: {file_path}")

    chunks = ingest_document(file_path)

    return {
        "file":    file_path,
        "chunks":  chunks,
        "message": f"Successfully ingested {chunks} chunks from '{file_path}'.",
    }