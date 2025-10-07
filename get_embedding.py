# download_embedding.py
from sentence_transformers import SentenceTransformer

def download_embedding_model():
    model_name = "BAAI/bge-m3"  # embedding mạnh và đa ngôn ngữ
    local_dir = "./embeddings/bge-m3"

    print(f"🔹 Đang tải model {model_name} ...")
    model = SentenceTransformer(model_name)
    model.save(local_dir)
    print(f"✅ Đã lưu model vào {local_dir}")

if __name__ == "__main__":
    download_embedding_model()
