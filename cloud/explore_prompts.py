#%%
import os
import pandas as pd
from utils import image_grid

song_name = 'spacetrain_1024' #@param {type:"string"}
res_height = 576 #@param
res_width = 1024 #@param

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                               torch_dtype=torch.float16,
                                               safety_checker=None,
                                               cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")


#%%
fp = os.path.join('prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

name_sel = 'solarpunk'
prompt = df_prompt['prompt'][name_sel]

# rows X cols of images. Reduce for speed and memory issues.
rows = 2
cols = 2

num_images = rows*cols

# Make new random seeds in a hacky way. TODO: probably a function to generate seeds without making a generator instance.

generator = torch.Generator(device="cuda")
seeds = [generator.seed() for i in range(num_images)]
generator = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

print("Prompt: {}".format(prompt))
print("Seeds: {}".format(seeds))

images = pipe(prompt, generator=generator, num_images_per_prompt=rows*cols, width=res_width, height=res_height).images

grid = image_grid(images, rows=rows, cols=cols)

grid
# %%


output_dir = os.path.join('output', 'explore', name_sel)
if not os.path.exists(output_dir): os.makedirs(output_dir)

grid.save(os.path.join(output_dir, 'image_grid.png'))

for i, image in enumerate(images):
    image.save(os.path.join(output_dir, 'image_{}.png'.format(i)))
# %%
