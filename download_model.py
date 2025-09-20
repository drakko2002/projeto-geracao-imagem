from huggingface_hub import snapshot_download
import os

# Baixa apenas os arquivos essenciais em fp16, pra não baixar
# os 18 gb do modelo completo em float32.
# Veja mais em https://huggingface.co/docs/huggingface_hub/how-to

print("Baixando modelo Stable Diffusion 2.1...")

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    local_dir="./model",
    allow_patterns=["*fp16*.safetensors", "*.json", "*.txt", "*.py", "*.onnx", "*.bin"],
    token=os.getenv("HUGGINGFACE_HUB_TOKEN")
)

print("Download completo! Modelo salvo em './model'")
