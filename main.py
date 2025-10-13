from fastapi import FastAPI, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch, faiss, pandas as pd
from docx import Document
import io, pickle, os, datetime, re

app = FastAPI(title="RAG Multi-file API", description="Upload nhiều file và hỏi LLM với prompt tối ưu")

# --------- Load local models ----------
llm_path = "./models/llm/llama2-7b-chat"
embed_path = "./models/embeddings/bge-m3"  # hoặc intfloat/multilingual-e5-large

tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="auto",
    torch_dtype="auto"
)
embed_model = SentenceTransformer(embed_path)

# --------- File paths ----------
FAISS_PATH = "index.faiss"
DOCS_PATH = "documents.pkl"

# --------- Initialize or load existing index ----------
dimension = embed_model.get_sentence_embedding_dimension()
if os.path.exists(FAISS_PATH) and os.path.exists(DOCS_PATH):
    print("🔹 Loading existing FAISS index and documents...")
    index = faiss.read_index(FAISS_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
else:
    print("🆕 Creating new FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    documents = []

# --------- Helpers ----------
def read_txt(file_bytes: bytes):
    return file_bytes.decode("utf-8", errors="ignore")

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

def smart_chunk(text, chunk_size=350):
    """Tách văn bản thành đoạn thông minh dựa trên tiêu đề / độ dài."""
    # tách theo tiêu đề hoặc dòng trống
    parts = re.split(r'\n(?=\d+\.)|\n(?=\(\d+\))|(?=\n[A-Z])', text)
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

def save_index_and_docs():
    faiss.write_index(index, FAISS_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print("💾 Đã lưu FAISS và documents thành công.")

# --------- API Models ----------
class Query(BaseModel):
    question: str
    max_new_tokens: int = 300
    debug: Optional[bool] = False  # nếu True -> trả thêm raw_answer và context

# --------- Upload nhiều file ----------
@app.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    uploaded_info = []
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
            uploaded_info.append({"file": file.filename, "status": "unsupported"})
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

        uploaded_info.append({"file": file.filename, "chunks": len(chunks)})

    save_index_and_docs()
    return {
        "status": "uploaded",
        "total_files": len(uploaded_info),
        "files": uploaded_info,
        "total_documents": len(documents)
    }

# --------- Ask question ----------
@app.post("/ask")
def ask_question(query: Query):
    if len(documents) == 0:
        return {"error": "Chưa có tài liệu nào, hãy upload trước."}

    q_emb = embed_model.encode([query.question])
    D, I = index.search(q_emb, k=8)  # lấy nhiều đoạn hơn
    retrieved = "\n\n".join([documents[i]["text"] for i in I[0]])

    # Prompt tối ưu tiếng Việt
    prompt = f"""
Bạn là một chuyên gia phân tích quân sự, hãy đọc kỹ NGỮ CẢNH sau đây và trả lời CÂU HỎI.
- Trích dẫn đúng thông tin trong tài liệu, không suy diễn.
- Nếu tài liệu có đoạn liên quan, hãy tổng hợp lại thành câu văn rõ ràng, súc tích.
- Nếu không có thông tin, chỉ cần trả lời: "Không tìm thấy thông tin liên quan."

--- NGỮ CẢNH ---
{retrieved}

--- CÂU HỎI ---
{query.question}

--- TRẢ LỜI (bằng tiếng Việt) ---
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=query.max_new_tokens,
        temperature=0.3,
        top_p=0.9
    )
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------- Làm sạch output --------
    clean_answer = raw_text.split("--- TRẢ LỜI")[-1]
    clean_answer = clean_answer.replace("(bằng tiếng Việt)", "")
    clean_answer = clean_answer.replace(":", "").strip()
    if "--- NGỮ CẢNH" in clean_answer:
        clean_answer = clean_answer.split("--- NGỮ CẢNH")[0].strip()

    # Kết quả đầu ra
    result = {"answer": clean_answer}

    if query.debug:
        result["context_used"] = retrieved[:600] + "..."
        result["raw_answer"] = raw_text

    return result

# --------- Reset toàn bộ dữ liệu ----------
@app.post("/reset_index")
def reset_index():
    global index, documents
    index = faiss.IndexFlatL2(dimension)
    documents = []
    if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
    if os.path.exists(DOCS_PATH): os.remove(DOCS_PATH)
    return {"status": "reset", "message": "Đã xóa toàn bộ dữ liệu."}
