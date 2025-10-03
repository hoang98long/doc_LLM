from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch, faiss

app = FastAPI()

# --- Load local models ---
llm_path = "./models/llm/llama2-7b-chat"
embed_path = "./models/embeddings/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="auto",
    torch_dtype="auto"
)
embed_model = SentenceTransformer(embed_path)

# FAISS DB
dimension = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
documents = []


# --- Schema ---
class Query(BaseModel):
    question: str
    max_new_tokens: int = 200


# --- Upload tài liệu dạng file txt ---
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    # chunk đơn giản theo dòng
    for line in content.split("\n"):
        if line.strip():
            emb = embed_model.encode([line])
            index.add(emb)
            documents.append(line.strip())
    return {"status": "uploaded", "chunks": len(documents)}


# --- Đặt câu hỏi ---
@app.post("/ask")
def ask_question(query: Query):
    if len(documents) == 0:
        return {"error": "Chưa có tài liệu nào, hãy upload trước."}

    q_emb = embed_model.encode([query.question])
    D, I = index.search(q_emb, k=1)
    context = documents[I[0][0]]

    prompt = f"""Bạn là trợ lý AI, hãy trả lời dựa trên tài liệu dưới đây.

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
