#%%
import os
import pandas as pd
from aa_utils.sd import image_grid, generate_latent, get_text_embed
from aa_utils.cloud import load_df_prompt, gen_pipe
import torch
from PIL import Image

import argparse
import json
import dotenv
import os

dotenv.load_dotenv()

# add arg for song name 

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--prompt_name', '-p', type=str, default="geo1")
parser.add_argument('--num_images', '-n', type=int, default=4)
args = parser.parse_args()
# args=None

song_name = args.song_name if args.song_name else 'escape'
num_images = args.num_images if args.num_images else 4


song_meta_dir = os.path.join(os.getenv('REPO_DIR'), 'song_meta', song_name)
# load json file with song settings
json_fp = os.path.join(song_meta_dir, 'tgen_settings.json')

with open(json_fp, 'r') as f:
    settings = json.load(f)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)




#%%

df_prompt = load_df_prompt(song_meta_dir)

with open(json_fp, 'r') as f:
    settings = json.load(f)

if 'mask_image' in settings:
    mask_image = Image.open(os.path.join('masks', settings['mask_image']))
    settings['pipe_kwargs']['image'] = mask_image    

# if 'prompt_name' not in args:
#     name_sel = 'geo1'
# else:
#     name_sel = args.prompt_name

default_prompt = 'geo1'
name_sel = args.prompt_name if args.prompt_name else default_prompt

prompt = df_prompt['prompt'][name_sel]

col_wrap = 2 
# rows X cols of images. Reduce for speed and memory issues.
rows = 2
cols = 2

def calculate_rows_cols(num_images, col_wrap):
    cols = col_wrap
    rows = num_images // cols
    if num_images % cols != 0:
        rows += 1
    return rows, cols

rows, cols = calculate_rows_cols(args.num_images, col_wrap)

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
    **settings['pipe_kwargs']
    ).images

grid = image_grid(images, rows=rows, cols=cols)

grid
# %%


output_dir = os.path.join('output', song_name, 'explore_images', name_sel)
if not os.path.exists(output_dir): os.makedirs(output_dir)

grid.save(os.path.join(output_dir, 'image_grid.png'))

for i, image in enumerate(images):
    image.save(os.path.join(output_dir, 'image_{}.png'.format(i)))
# %%
