# Retriever logic
# backend/app/rag/retriever.py

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config.settings import settings
from app.logger.logger import logger


# ── Embeddings (same model as ingest — must match) ────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)


def load_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk.
    Raises FileNotFoundError if no documents have been ingested yet.
    """
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{settings.FAISS_INDEX_PATH}'. "
            "Please ingest documents first via POST /api/rag/ingest."
        )

    logger.info(f"[retriever] Loading FAISS index from {settings.FAISS_INDEX_PATH}")
    return FAISS.load_local(
        settings.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(query: str, k: int = None) -> list[Document]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Args:
        query : user's question
        k     : number of chunks to return (defaults to settings.RAG_TOP_K)

    Returns:
        List of LangChain Document objects with page_content + metadata
    """
    k = k or settings.RAG_TOP_K

    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    logger.info(f"[retriever] Retrieving top {k} chunks for: '{query}'")
    docs = retriever.invoke(query)
    logger.info(f"[retriever] Retrieved {len(docs)} chunks.")

    return docs


def format_context(docs: list[Document]) -> str:
    """
    Convert retrieved Document chunks into a clean numbered
    string block to inject into the LLM prompt.
    """
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        ref    = f"{source} p.{page}" if page != "" else source
        context_parts.append(
            f"[{i}] (Source: {ref})\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_parts)