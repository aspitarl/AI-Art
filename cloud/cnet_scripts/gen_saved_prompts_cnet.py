# %% [markdown]
# This is a notebook to generate all the prompts and seeds in the prompts google sheet

# %%

song_name = 'cycle_mask' #@param {type:"string"}
res_height = 576 #@param
res_width = 1024 #@param
seed_delimiter = ","

import os
import pandas as pd

from aa_utils.sd import generate_latent

from PIL import Image
mask_image = Image.open(os.path.join('output', song_name, "cyclist_side.png"))

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'
output_folder = os.path.join('output', song_name, 'prompt_images')
if not os.path.exists(output_folder): os.makedirs(output_folder)

fp = os.path.join('prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')
df_prompt = df_prompt.dropna(how='any', subset=['prompt', 'seeds'])


# %%


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None
)

# if one wants to disable `tqdm`
# https://github.com/huggingface/diffusers/issues/1786
pipe.set_progress_bar_config(disable=True)

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()





# %% [markdown]
# # Iterate through prompts and seeds, outputting an image for both

# %%
for name, row in df_prompt.iterrows():
  seeds = row['seeds'].split(seed_delimiter)
  seeds = [s.strip() for s in seeds]
  print(seeds)

# %%
device = "cuda"
generator = torch.Generator(device=device)


#TODO: replace below
width = res_width
height = res_height

skip_existing = True

for name, row in df_prompt.iterrows():
  seeds = row['seeds'].split(seed_delimiter)
  seeds = [s.strip() for s in seeds]

  prompt = row['prompt']
  guidance_scale = float(row['guidance_scale'])

  for seed in seeds:
    output_fn = "{}_{}.png".format(name, seed)

    if os.path.exists(os.path.join(output_folder, output_fn)):
      if skip_existing:
        print("{} already exists, skipping".format(output_fn))
        continue

    generator.manual_seed(int(seed))

    latent = generate_latent(generator, seed, pipe, height // 8, width // 8)

    with torch.autocast(device):
      images = pipe(
          prompt,
          guidance_scale=guidance_scale,
          latents = latent,
          width=width,
          height=height,
          control_guidance_start=0.1,
          control_guidance_end=0.6,          
          controlnet_conditioning_scale=0.8,
          image=mask_image
      )



    output_image = images.images[0]

    output_image.save(os.path.join(output_folder, output_fn))

# %%



