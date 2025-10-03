from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import torch

# -----------------------
# 1. Khởi tạo FastAPI
# -----------------------
app = FastAPI()

# -----------------------
# 2. Load LLM (3B-7B)
# -----------------------
llm_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    device_map="auto",
    torch_dtype="auto"
)

# -----------------------
# 3. Load embedding model
# -----------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index (vector DB)
dimension = 384  # Kích thước embedding của all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
documents = []  # Lưu lại text gốc theo id

# -----------------------
# 4. API models
# -----------------------
class Document(BaseModel):
    text: str

class Query(BaseModel):
    question: str
    max_new_tokens: int = 200


# -----------------------
# 5. Endpoint: Upload document
# -----------------------
@app.post("/upload_doc")
def upload_document(doc: Document):
    # Tạo embedding cho tài liệu
    embedding = embed_model.encode([doc.text])
    index.add(embedding)  # Thêm vào FAISS
    documents.append(doc.text)
    return {"status": "uploaded", "doc_id": len(documents) - 1}


# -----------------------
# 6. Endpoint: Ask question
# -----------------------
@app.post("/ask")
def ask_question(query: Query):
    # Encode câu hỏi thành embedding
    q_emb = embed_model.encode([query.question])

    # Lấy top-1 đoạn văn bản liên quan
    D, I = index.search(q_emb, k=1)
    context = documents[I[0][0]] if len(documents) > 0 else ""

    # Prompt cho LLM
    prompt = f"""
    Bạn là trợ lý AI. Hãy trả lời câu hỏi dựa trên tài liệu sau:

    TÀI LIỆU:
    {context}

    CÂU HỎI: {query.question}

    TRẢ LỜI:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "context": context}
