from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Chatbot with RAG")

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}
