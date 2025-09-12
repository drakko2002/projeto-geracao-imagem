from diffusers import DiffusionPipeline
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]