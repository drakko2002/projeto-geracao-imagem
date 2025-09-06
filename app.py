import os
from diffusers import StableDiffusionPipeline

# pega token do ambiente
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    use_auth_token=token,
    torch_dtype="auto"
).to("cpu")

image = pipe("um gato astronauta explorando marte").images[0]
image.save("saida.png")

print("Imagem gerada e salva como saida.png")
