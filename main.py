from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch, faiss, pandas as pd
from docx import Document
import io, os, pickle, datetime, re

app = FastAPI(
    title="RAG QA System",
    description="Upload tài liệu (txt, docx, xlsx) và hỏi đáp bằng LLM",
)

# ================== MODEL LOADING ==================
LLM_PATH = "./models/llm/llama2-7b-chat"          # model LLM local
EMBED_PATH = "./models/embeddings/bge-m3"         # model embedding local

print("Đang load mô hình...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
embed_model = SentenceTransformer(EMBED_PATH)

# ================== FAISS SETUP ==================
FAISS_PATH = "index.faiss"
DOCS_PATH = "documents.pkl"

dimension = embed_model.get_sentence_embedding_dimension()
if os.path.exists(FAISS_PATH) and os.path.exists(DOCS_PATH):
    print("🔹 Đang load FAISS index và tài liệu đã lưu...")
    index = faiss.read_index(FAISS_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
else:
    print("🆕 Tạo mới FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    documents = []

# ================== HELPER FUNCTIONS ==================
def read_txt(file_bytes: bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def read_docx(file_bytes: bytes):
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_xlsx(file_bytes: bytes):
    excel = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    out = []
    for sheet_name, df in excel.items():
        out.append(f"Sheet: {sheet_name}")
        out.append(df.to_string(index=False))
    return "\n".join(out)

def smart_chunk(text, chunk_size=350):
    """Chia văn bản thành các đoạn hợp lý."""
    parts = re.split(r"\n{2,}", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        words = p.split()
        if len(words) > chunk_size:
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i+chunk_size]))
        else:
            chunks.append(p)
    return chunks

def save_index():
    faiss.write_index(index, FAISS_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print("💾 Đã lưu FAISS và documents.")

# ================== API SCHEMAS ==================
class Query(BaseModel):
    question: str
    max_new_tokens: int = 250
    debug: Optional[bool] = False

# ================== API ROUTES ==================

@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        ext = file.filename.split(".")[-1].lower()
        content = await file.read()

        if ext == "txt":
            text = read_txt(content)
        elif ext == "docx":
            text = read_docx(content)
        elif ext in ["xls", "xlsx"]:
            text = read_xlsx(content)
        else:
            uploaded.append({"file": file.filename, "status": "unsupported"})
            continue

        chunks = smart_chunk(text)
        embeddings = embed_model.encode(chunks)
        index.add(embeddings)

        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for chunk in chunks:
            documents.append({
                "file": file.filename,
                "text": chunk,
                "upload_time": upload_time
            })

        uploaded.append({"file": file.filename, "chunks": len(chunks)})

    save_index()
    return {
        "status": "uploaded",
        "files": uploaded,
        "total_documents": len(documents)
    }

@app.post("/ask")
def ask(query: Query):
    if len(documents) == 0:
        return {"error": "Chưa có tài liệu nào. Hãy upload trước."}

    # ========== Lấy context ==========
    q_emb = embed_model.encode([query.question])
    D, I = index.search(q_emb, k=4)
    context = "\n\n".join([documents[i]["text"] for i in I[0]])

    # ========== Prompt ==========
    prompt = f"""
Bạn là một chuyên gia phân tích. Hãy đọc kỹ NGỮ CẢNH và trả lời CÂU HỎI.
- Trả lời bằng tiếng Việt, ngắn gọn, chính xác, không lặp lại từ vô nghĩa.
- Nếu không có thông tin, nói rõ: "Không tìm thấy thông tin liên quan."

NGỮ CẢNH:
{context}

CÂU HỎI:
{query.question}

TRẢ LỜI:
"""

    # ========== Sinh câu trả lời ==========
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_new_tokens,
        temperature=0.1,
        top_p=0.8
    )

    raw_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # ========== Làm sạch ==========
    # Chỉ lấy phần sau "TRẢ LỜI:"
    if "TRẢ LỜI" in raw_text:
        answer = raw_text.split("TRẢ LỜI")[-1].replace(":", "").strip()
    else:
        answer = raw_text.strip()

    # Loại bỏ chuỗi rác (ký tự lặp)
    answer = re.sub(r"([^\w\s])\1{3,}", "", answer)
    answer = re.sub(r"(\b\w+\b)(?:\s+\1){2,}", r"\1", answer)

    result = {"answer": answer}

    if query.debug:
        result["context_used"] = context[:600] + "..."
        result["raw_answer"] = raw_text

    return result

@app.post("/reset_index")
def reset_index():
    global index, documents
    index = faiss.IndexFlatL2(dimension)
    documents = []
    if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
    if os.path.exists(DOCS_PATH): os.remove(DOCS_PATH)
    return {"status": "reset", "message": "Đã xóa toàn bộ dữ liệu."}

# ================== RUN INFO ==================
@app.get("/")
def home():
    return {"message": "RAG API is running. Use /docs to test."}
