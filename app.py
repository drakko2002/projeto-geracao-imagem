import torch
from diffusers import StableDiffusionPipeline
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_path = "./model"
if not os.path.exists(model_path):
    raise ValueError(f"Model path {model_path} does not exist. Please run download_model.py first.")

pipe = StableDiffusionPipeline.from_pretrained(
    "./model",
    torch_dtype=torch.float16,
    use_auth_token=token,
    local_files_only=True #Força o uso apenas dos arquivos locais, caso detectados.
).to("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./outputs", exist_ok=True)
# Ativa attention slicing para economizar memória
os.environ["DIFFUSERS_ATTENTION_SLICING"] = "1" # Ativa attention slicing para economizar memória
os.environ["DIFFUSERS_USE_MEMORY_EFFICIENT_ATTENTION"] = "1" # Ativa atenção eficiente em memória
os.environ["HF_HUB_OFFLINE"] = "1" # Força o uso offline do hub

pipe.enable_attention_slicing() # Ativa attention slicing para economizar memória
pipe.enable_xformers_memory_efficient_attention() # Ativa atenção eficiente em memória

image = pipe("um gato astronauta explorando marte").images[0]
image.save("saida.png")

print("Imagem gerada e salva como saida.png")
