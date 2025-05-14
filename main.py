from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Nếu bạn có thêm file riêng xử lý tài liệu, bạn cũng import ở đây
from qa_engine import search_answer, upload_document

app = FastAPI()

# Bật CORS cho tất cả các domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc liệt kê domain cụ thể ["https://your-frontend.up.railway.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cho /upload
class UploadRequest(BaseModel):
    content: str

# Model cho /ask
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
