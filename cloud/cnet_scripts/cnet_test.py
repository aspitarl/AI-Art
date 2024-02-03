#%%
from PIL import Image
import os

song_folder = "output/cycle"

image = Image.open(os.path.join(song_folder, "bikemask.png"))

# %%

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None
)

#%%
controlnet

#%%


from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()

#%%

generator = torch.manual_seed(1)


prompt = "a man in an neon green hazmat suit riding a bike, an abandoned highway with broken cars, bleak gray, post-apocalyptic, bird's eye view, far away"

out_image = pipe(
    prompt, 
    num_inference_steps=20, 
    generator=generator, 
    image=image,
    control_guidance_start=0.1,
    control_guidance_end=0.4,
).images[0]


out_image.save('test.png')