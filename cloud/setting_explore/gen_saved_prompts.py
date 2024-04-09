# %% [markdown]
# This is a notebook to generate all the prompts and seeds in the prompts google sheet

# %%


import os
import pandas as pd
import json
import dotenv
import argparse
import torch
from PIL import Image
import itertools

from aa_utils.sd import generate_latent, get_text_embed
from aa_utils.cloud import load_df_prompt, gen_pipe

dotenv.load_dotenv()

repo_dir = os.getenv('REPO_DIR')

# add arg for song name 

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--output_dir', '-o', type=str, help='The output directory for the images', default='test1')
args = parser.parse_args()
song_name = args.song_name

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'
output_basedir = os.path.join('output', song_name, args.output_dir)
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

dir_prompt_data = os.path.join(repo_dir, 'cloud', 'prompt_data', song_name)
song_meta_dir = os.path.join(repo_dir, 'song_meta', song_name)

# load json file with song settings
json_fp = os.path.join(song_meta_dir, 'tgen_settings.json')

with open(json_fp, 'r') as f:
    settings = json.load(f)

df_prompt = load_df_prompt(song_meta_dir)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)

if 'mask_image' in settings:
    mask_image = Image.open(os.path.join(os.getenv('REPO_DIR'), 'cloud', 'masks', settings['mask_image']))
    settings['pipe_kwargs']['image'] = mask_image     


# %% [markdown]
# # Iterate through prompts and seeds, outputting an image for both

# %%

if 'seed_delimiter' not in settings:
    seed_delimiter = ', '
else:
    seed_delimiter = settings['seed_delimiter']

for name, row in df_prompt.iterrows():
    seeds = row['seeds'].split(seed_delimiter)
    seeds = [s.strip() for s in seeds]
    seeds = [int(s) for s in seeds]


# %%
device = "cuda"
generator = torch.Generator(device=device)

prompt_sel = ['geo1', 'geo_simple1', 'portal2', 'geom1']
# prompt_sel = ['ridin2', 'elatrain2', 'mushrooms2', 'flyin2']
df_prompt = df_prompt.loc[prompt_sel]

skip_existing = True

cnet_vals = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0]
# masks = ['window_net', 'window_net_blur']
masks = ['circle', 'circle0', 'circle2']



for name, row in df_prompt.iterrows():


    seeds = [seeds[0]] # Keep only the first seed 

    prompt = row['prompt']
    guidance_scale = float(row['guidance_scale'])

    # Use itertools.product to generate all combinations of seeds and cnet_vals
    for seed, cnet_val, mask_name in itertools.product(seeds, cnet_vals, masks):
        print("cnet_val: {}".format(cnet_val))
        settings['pipe_kwargs']['controlnet_conditioning_scale'] = cnet_val
        settings['pipe_kwargs']['image'] = Image.open(os.path.join(os.getenv('REPO_DIR'), 'cloud', 'masks', mask_name + '.png'))

        cnet_val_str = str(cnet_val).replace('.', 'p')
        mask_str = mask_name.replace('_', '')
        output_fn = "{}_{}_{}_{}.png".format(name, seed, cnet_val_str, mask_str)

        if os.path.exists(os.path.join(output_basedir, output_fn)):
            if skip_existing:
                print("{} already exists, skipping".format(output_fn))
                continue

        latent = generate_latent(generator, seed, pipe, settings['res_height'] // 8, settings['res_width'] // 8)
        text_embed = get_text_embed(prompt, pipe)

        with torch.autocast(device):
            images = pipe(
                prompt_embeds=text_embed,
                guidance_scale=guidance_scale,
                latents = latent,
                **settings['pipe_kwargs']
            )

        output_image = images.images[0]

        output_image.save(os.path.join(output_basedir, output_fn))