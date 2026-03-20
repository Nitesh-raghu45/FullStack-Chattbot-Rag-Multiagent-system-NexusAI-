from fastapi import APIRouter
from app.services.chat_service import chat_response
from app.services.rag_service import rag_response

router = APIRouter()

@router.post("/chat")
def chat(data: dict):
    return {"response": chat_response(data["message"])}

@router.post("/rag")
def rag(data: dict):
    return {"response": rag_response(data["query"])}
