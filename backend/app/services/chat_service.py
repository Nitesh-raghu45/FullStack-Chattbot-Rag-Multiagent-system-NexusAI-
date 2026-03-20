from app.chatbot.service import get_chat_response

def chat_response(message: str):
    return get_chat_response(message)
