from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class RAGRequest(BaseModel):
    query: str
