#%%
import os
import pandas as pd
from aa_utils.sd import image_grid

import argparse
import json
import dotenv
import os

dotenv.load_dotenv()

repo_dir = os.getenv('REPO_DIR')

# add arg for song name 

# parser = argparse.ArgumentParser(description='Generate transitions between prompts')
# parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
# args = parser.parse_args()
# song_name = args.song_name
song_name = 'escape'


song_meta_dir = os.path.join(repo_dir, 'song_meta', song_name)
# load json file with song settings
json_fp = os.path.join(song_meta_dir, 'tgen_settings.json')

with open(json_fp, 'r') as f:
    settings = json.load(f)

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(settings['model_string'],
                                               torch_dtype=torch.float16,
                                               safety_checker=None,
                                               cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")


#%%
fp = os.path.join(song_meta_dir, 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

with open(json_fp, 'r') as f:
    settings = json.load(f)

name_sel = 'geo1'
prompt = df_prompt['prompt'][name_sel]

# rows X cols of images. Reduce for speed and memory issues.
rows = 2
cols = 2

num_images = rows*cols

# Make new random seeds in a hacky way. TODO: probably a function to generate seeds without making a generator instance.

generator = torch.Generator(device="cuda")
seeds = [generator.seed() for i in range(num_images)]
# truncate seeds to 4 digits
seeds = [int(str(seed)[:4]) for seed in seeds]

generator = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

print("Prompt: {}".format(prompt))
print("Seeds: {}".format(seeds))

images = pipe(
    prompt, 
    generator=generator, 
    num_images_per_prompt=rows*cols, 
    width=settings['res_width'], 
    height=settings['res_height'],
    num_inference_steps=settings['inference_steps']
    ).images

grid = image_grid(images, rows=rows, cols=cols)

grid
# %%


output_dir = os.path.join('output', 'explore', name_sel)
if not os.path.exists(output_dir): os.makedirs(output_dir)

grid.save(os.path.join(output_dir, 'image_grid.png'))

for i, image in enumerate(images):
    image.save(os.path.join(output_dir, 'image_{}.png'.format(i)))
# %%
