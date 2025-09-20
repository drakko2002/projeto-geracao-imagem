from huggingface_hub import snapshot_download
import os

# Carregar variáveis do arquivo .env
def load_env():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

# Baixa apenas os arquivos essenciais em fp16, pra não baixar
# os 18 gb do modelo completo em float32.
# Veja mais em https://huggingface.co/docs/huggingface_hub/how-to

print("Baixando modelo Stable Diffusion 2.1...")

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    print("Erro: Token HuggingFace não encontrado!")
    print("Configure o arquivo .env com seu token do HuggingFace")
    print("Obtenha em: https://huggingface.co/settings/tokens")
    exit(1)

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    local_dir="./model",
    allow_patterns=["*fp16*.safetensors", "*.json", "*.txt", "*.py", "*.onnx", "*.bin"],
    token=token
)

print("Download completo! Modelo salvo em './model'")
