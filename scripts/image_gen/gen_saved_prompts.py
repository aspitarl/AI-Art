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

from aa_utils.sd import generate_latent, get_text_embed
from aa_utils.cloud import load_df_prompt, gen_pipe, gen_pipe_kwargs_static

dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--setting_name', '-sn', type=str, default='default', nargs='?', help='Name of top-level key in settings json')
args = parser.parse_args()
song_name = args.song_name

output_basedir = os.path.join(os.getenv('media_dir'), "{}".format(song_name), 'prompt_images')
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

dir_transition_meta = os.path.join(os.getenv('media_dir'), 'transition_meta', song_name)
song_meta_dir = os.path.join(os.getenv('meta_dir'), song_name)

# load json file with song settings
with open(os.path.join(song_meta_dir, 'tgen_settings.json'), 'r') as f:
    settings = json.load(f)[args.setting_name]

seed_delimiter = settings.get('seed_delimiter', ', ')
df_prompt = load_df_prompt(song_meta_dir, seed_delimiter)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)

# %% [markdown]
# # Iterate through prompts and seeds, outputting an image for both

# %%

# %%
device = "cuda"
generator = torch.Generator(device=device)


skip_existing = True

for name, row in df_prompt.iterrows():

    seeds = row['seeds']

    prompt = row['prompt']

    pipe_kwargs = gen_pipe_kwargs_static(row, pipe_name, song_name)
    settings['pipe_kwargs'].update(pipe_kwargs)

    for seed in seeds:
        output_fn = "{}_{}.png".format(name, seed)

        if os.path.exists(os.path.join(output_basedir, output_fn)):
            if skip_existing:
                print("{} already exists, skipping".format(output_fn))
                continue

        latent = generate_latent(generator, seed, pipe, settings['res_height'] // 8, settings['res_width'] // 8)
        text_embed = get_text_embed(prompt, pipe)

        with torch.autocast(device):
            images = pipe(
                prompt_embeds=text_embed,
                latents = latent,
                **settings['pipe_kwargs']
            )

        output_image = images.images[0]

        output_image.save(os.path.join(output_basedir, output_fn))

# %%



