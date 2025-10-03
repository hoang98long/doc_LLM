from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"  # thay bằng model bạn muốn

print("Đang tải tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Đang tải model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",   # tự động dùng GPU/CPU
    torch_dtype="auto"
)