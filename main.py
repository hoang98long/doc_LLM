from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch, faiss, pandas as pd
from docx import Document
import io

app = FastAPI()

# --------- Load local models ----------
llm_path = "./models/llm/llama2-7b-chat"
embed_path = "./models/embeddings/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="auto",
    torch_dtype="auto"
)
embed_model = SentenceTransformer(embed_path)

# --------- FAISS index ----------
dimension = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
documents = []


# --------- Helpers ----------
def read_txt(file_bytes: bytes):
    return file_bytes.decode("utf-8")

def read_docx(file_bytes: bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_xlsx(file_bytes: bytes):
    excel = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    text_chunks = []
    for sheet_name, df in excel.items():
        text_chunks.append(f"Sheet: {sheet_name}")
        text_chunks.append(df.to_string(index=False))
    return "\n".join(text_chunks)


def chunk_text(text, chunk_size=300):
    """Chia tài liệu thành đoạn nhỏ 300 từ để dễ tìm kiếm"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# --------- API models ----------
class Query(BaseModel):
    question: str
    max_new_tokens: int = 300


# --------- Upload tài liệu đa định dạng ----------
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    content = await file.read()

    if ext == "txt":
        text = read_txt(content)
    elif ext == "docx":
        text = read_docx(content)
    elif ext in ["xls", "xlsx"]:
        text = read_xlsx(content)
    else:
        return {"error": f"Định dạng {ext} chưa được hỗ trợ."}

    chunks = chunk_text(text)
    for c in chunks:
        emb = embed_model.encode([c])
        index.add(emb)
        documents.append(c)

    return {"status": "uploaded", "chunks": len(chunks), "file": file.filename}


# --------- Prompt engineering + Ask ----------
@app.post("/ask")
def ask_question(query: Query):
    if len(documents) == 0:
        return {"error": "Chưa có tài liệu nào, hãy upload trước."}

    q_emb = embed_model.encode([query.question])
    D, I = index.search(q_emb, k=3)
    retrieved = "\n\n".join([documents[i] for i in I[0]])

    # Prompt engineering — hướng dẫn rõ ràng hơn
    prompt = f"""
Bạn là một trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu. 
Hãy sử dụng nội dung trong phần "Ngữ cảnh" bên dưới để trả lời chính xác, ngắn gọn, bằng tiếng Việt.
Nếu không tìm thấy thông tin trong tài liệu, hãy nói "Tôi không tìm thấy thông tin trong tài liệu."

Ngữ cảnh:
{retrieved}

Câu hỏi: {query.question}

Câu trả lời:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "context_used": retrieved[:500]}  # cắt ngắn context cho dễ đọc
