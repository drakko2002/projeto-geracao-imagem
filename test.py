import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "models/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

image = pipe("um gato astronauta explorando marte").images[0]
image.save("gato_marte.png")
