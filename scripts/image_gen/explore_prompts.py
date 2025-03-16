#%%
import os
import pandas as pd
from aa_utils.sd import image_grid, generate_latent, get_text_embed
from aa_utils.cloud import gen_pipe, gen_pipe_kwargs_static
from aa_utils.fileio import load_df_prompt
from aa_utils.fileio import load_settings_json
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
parser.add_argument('--prompt_names', '-p', type=str, required=True, nargs='+', help='List of prompt names to generate images for')
parser.add_argument('--setting_name', '-sn', type=str, default='default', nargs='?', help='Name of top-level key in settings json')
parser.add_argument('--num_images', '-n', type=int, default=4)
# add a all sub prompts flag, stores True to an argument that will iterate through all subprompts (promt_base_substr)
parser.add_argument('--all_subprompts', '-a', action='store_true', help='Generate images for all subprompts')
args = parser.parse_args()
# args=None

#%%
song_meta_dir = os.path.join(os.getenv('meta_dir'), args.song_name)

settings = load_settings_json(song_meta_dir, setting_name=args.setting_name)

df_prompt = load_df_prompt(song_meta_dir, seed_delimiter=settings['seed_delimiter'])

pipe = gen_pipe(settings)

col_wrap = 2 
# rows X cols of images. Reduce for speed and memory issues.
rows = 2
cols = 3

def calculate_rows_cols(num_images, col_wrap):
    cols = col_wrap
    rows = num_images // cols
    if num_images % cols != 0:
        rows += 1
    return rows, cols

rows, cols = calculate_rows_cols(args.num_images, col_wrap)

# Make new random seeds in a hacky way. TODO: probably a function to generate seeds without making a generator instance.

generator = torch.Generator(device="cuda")
seeds = [generator.seed() for i in range(args.num_images)]
# truncate seeds to 4 digits
seeds = [int(str(seed)[:4]) for seed in seeds]

generator = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

# Make a dataframe to iterate over, either all subprompts or just the ones specified
if args.all_subprompts:
    prompt_names = df_prompt.index
    prompt_names = [prompt_name for prompt_name in prompt_names if prompt_name.startswith(args.prompt_names[0])]
else:
    prompt_names = args.prompt_names

print("Iterating over prompts: {}\n".format(prompt_names))

for prompt_name in prompt_names:
    prompt = df_prompt['prompt'][prompt_name]

    pipe_kwargs = gen_pipe_kwargs_static(df_prompt.loc[prompt_name], settings['pipe_name'], args.song_name)
    settings['pipe_kwargs'].update(pipe_kwargs)

    print("\nPrompt name: {}".format(prompt_name))
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

    output_dir = os.path.join(os.getenv('media_dir'), args.song_name, 'explore_images', prompt_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    grid.save(os.path.join(output_dir, 'image_grid.png'))

    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, 'image_{}.png'.format(i)))
# %%
