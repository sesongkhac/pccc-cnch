# qa_engine.py

import requests
import config
import numpy as np

documents = []

def add_document(text):
    documents.append(text)

def embed_text(text):
    headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_TOKEN}"}
    response = requests.post(
        config.HUGGINGFACE_EMBEDDING_API_URL,
        headers=headers,
        json={"inputs": text}
    )
    return response.json()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_documents(question, top_k=3):
    question_embedding = embed_text(question)
    if not question_embedding or "error" in question_embedding:
        return ["Không tìm thấy embedding."]
    
    similarities = []
    for doc in documents:
        doc_embedding = embed_text(doc)
        if doc_embedding and "error" not in doc_embedding:
            sim = cosine_similarity(question_embedding, doc_embedding)
            similarities.append((sim, doc))
    
    similarities.sort(reverse=True)
    top_docs = [doc for _, doc in similarities[:top_k]]
    return top_docs
