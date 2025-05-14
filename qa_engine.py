from typing import List
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

documents: List[str] = []
document_embeddings = None

def upload_document(content: str):
    global documents, document_embeddings
    documents.append(content)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    return {"status": "uploaded", "total_documents": len(documents)}

def search_answer(question: str):
    if not documents:
        return "Không có tài liệu nào để tìm kiếm."

    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, document_embeddings)[0]
    best_idx = int(scores.argmax())
    return documents[best_idx]