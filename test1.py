import torch
import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set")
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
    
image.save("astronaut_rides_horse.png")
