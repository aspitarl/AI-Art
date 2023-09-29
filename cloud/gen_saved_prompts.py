# %% [markdown]
# This is a notebook to generate all the prompts and seeds in the prompts google sheet

# %%
song_name = 'spacetrain_1024' #@param {type:"string"}
res_height = 576 #@param
res_width = 1024 #@param
seed_delimiter = ","

import os
import pandas as pd

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'
output_folder = os.path.join('output', song_name, 'prompt_images')
if not os.path.exists(output_folder): os.makedirs(output_folder)

fp = os.path.join('prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')
df_prompt = df_prompt.dropna(how='any')


# %%
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                               torch_dtype=torch.float16,
                                               safety_checker=None,
                                               cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")



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

    latent = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )

    with torch.autocast(device):
      images = pipe(
          prompt,
          guidance_scale=guidance_scale,
          latents = latent,
          width=width,
          height=height
      )

    output_image = images.images[0]

    output_image.save(os.path.join(output_folder, output_fn))

# %%



