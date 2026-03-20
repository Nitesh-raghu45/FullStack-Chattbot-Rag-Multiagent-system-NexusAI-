from app.rag.service import get_rag_response

def rag_response(query: str):
    return get_rag_response(query)
