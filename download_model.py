from huggingface_hub import snapshot_download

# Baixa apenas os arquivos essenciais em fp16, pra nao baixar
#os 18 gb do modelo completo em float32.
# Veja mais em https://huggingface.co/docs/huggingface_hub/how-to
snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    local_dir="models/stable-diffusion-2-1",
    allow_patterns=["*fp16*.safetensors", "*.json", "*.txt", "*.py"]
)
