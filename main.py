from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch, faiss

app = FastAPI()

# Load từ local dir (không cần token)
llm_path = "./models/llm/llama2-7b-chat"
embed_path = "./models/embeddings/all-MiniLM-L6-v2"

# LLM
tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="auto",
    torch_dtype="auto"
)

# Embedding model
embed_model = SentenceTransformer(embed_path)

# FAISS index
dimension = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
documents = []

class Document(BaseModel):
    text: str

class Query(BaseModel):
    question: str

@app.post("/upload_doc")
def upload_doc(doc: Document):
    emb = embed_model.encode([doc.text])
    index.add(emb)
    documents.append(doc.text)
    return {"status": "ok", "doc_id": len(documents)-1}

@app.post("/ask")
def ask(query: Query):
    q_emb = embed_model.encode([query.question])
    D, I = index.search(q_emb, k=1)
    context = documents[I[0][0]] if len(documents) > 0 else ""

    prompt = f"Context: {context}\nQuestion: {query.question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer, "context": context}
