from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qa_engine import upload_document, search_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho tất cả domain frontend kết nối
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadRequest(BaseModel):
    content: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Simple Document Q&A Bot running."}

@app.post("/upload")
async def upload(req: UploadRequest):
    result = upload_document(req.content)
    return {"message": "Document uploaded successfully.", "result": result}

@app.post("/ask")
async def ask(req: QuestionRequest):
    answer = search_answer(req.question)
    return {"answer": answer}
