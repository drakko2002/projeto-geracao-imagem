import torch
from diffusers import StableDiffusionPipeline
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    use_auth_token=token
).to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_attention_slicing()

image = pipe("um gato astronauta explorando marte").images[0]
image.save("saida.png")

print("Imagem gerada e salva como saida.png")
