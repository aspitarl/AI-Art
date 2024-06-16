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
from aa_utils.cloud import load_df_prompt, gen_pipe

dotenv.load_dotenv()

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('setting_name', type=str, default='default', nargs='?', help='Name of top-level key in settings json')
args = parser.parse_args()
song_name = args.song_name
setting_name = args.setting_name

output_basedir = os.path.join(os.getenv('media_dir'), "{}".format(song_name), 'prompt_images')
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

dir_transition_meta = os.path.join(os.getenv('media_dir'), 'transition_meta', song_name)
song_meta_dir = os.path.join(os.getenv('meta_dir'), song_name)

# load json file with song settings
json_fp = os.path.join(song_meta_dir, 'tgen_settings.json')
with open(json_fp, 'r') as f:
    settings = json.load(f)

settings = settings[setting_name]

df_prompt = load_df_prompt(song_meta_dir)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)

if 'mask_image' in settings:
    mask_image = Image.open(os.path.join('masks', settings['mask_image']))
    settings['pipe_kwargs']['image'] = mask_image     


# %% [markdown]
# # Iterate through prompts and seeds, outputting an image for both

# %%

if 'seed_delimiter' not in settings:
    seed_delimiter = ', '
else:
    seed_delimiter = settings['seed_delimiter']


# %%
device = "cuda"
generator = torch.Generator(device=device)


skip_existing = True

for name, row in df_prompt.iterrows():

    seeds = row['seeds'].split(seed_delimiter)
    seeds = [s.strip() for s in seeds]
    seeds = [int(s) for s in seeds]

    prompt = row['prompt']
    guidance_scale = float(row['guidance_scale'])

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
                guidance_scale=guidance_scale,
                latents = latent,
                **settings['pipe_kwargs']
            )

        output_image = images.images[0]

        output_image.save(os.path.join(output_basedir, output_fn))

# %%



