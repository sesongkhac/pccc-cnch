# main.py

from fastapi import FastAPI, Request
from qa_engine import add_document, search_documents
import requests
import config

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Simple Document Q&A Bot running."}

@app.post("/upload")
async def upload_document(request: Request):
    data = await request.json()
    content = data.get("content", "")
    if content:
        add_document(content)
        return {"status": "Document added."}
    else:
        return {"status": "No content received."}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return {"answer": "Bạn chưa nhập câu hỏi."}
    
    contexts = search_documents(question)
    context = "\n".join(contexts)

    headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}"}
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(
        config.HUGGINGFACE_ANSWERING_API_URL,
        headers=headers,
        json=payload
    )

    if response.ok:
        return {"answer": response.json().get("answer", "Không tìm thấy câu trả lời.")}
    else:
        return {"answer": "Không thể lấy câu trả lời từ mô hình."}
