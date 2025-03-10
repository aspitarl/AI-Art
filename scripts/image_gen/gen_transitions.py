# %%
import os
import pandas as pd
import numpy as np
from IPython.display import clear_output
from aa_utils.sd import generate_latent, get_text_embed, slerp
from aa_utils.cloud import load_df_transitions,load_df_prompt, gen_pipe, gen_pipe_kwargs_transition
import torch
import dotenv; dotenv.load_dotenv()
import argparse
import json
from PIL import Image

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--setting_name', '-sn', type=str, default='default', nargs='?', help='Name of top-level key in settings json')
args = parser.parse_args()
song_name = args.song_name
setting_name = args.setting_name

output_basedir = os.path.join(os.getenv('media_dir'), "{}".format(song_name), 'transition_images')
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

dir_transition_meta = os.path.join(os.getenv('media_dir'), song_name, 'transition_meta')
song_meta_dir = os.path.join(os.getenv('meta_dir'), song_name)

# load json file with song settings
with open(os.path.join(song_meta_dir, 'tgen_settings.json'), 'r') as f:
    settings = json.load(f)[args.setting_name]

seed_delimiter = settings.get('seed_delimiter', ', ')
df_prompt = load_df_prompt(song_meta_dir, seed_delimiter)

df_transitions = load_df_transitions(dir_transition_meta)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)

# %%
skip_existing = True

generator = torch.Generator(device="cuda")

max_seed_characters = 4 # Take the first few numbers of the seed for the name
num_interpolation_steps = settings['interpolation_steps']


T = np.linspace(0.0, 1.0, num_interpolation_steps)

from tqdm import tqdm

for i_row, (idx, row) in enumerate(df_transitions.iterrows()):
    clear_output(wait=True)

    output_name = row.name

    output_dir = os.path.join(output_basedir, output_name)

    if os.path.exists(output_dir):
        if skip_existing:
            print("{} already exists, skipping".format(output_name))
            continue
        else:
            print("{} already exists, deleting images".format(output_name))
            for fn in os.listdir(output_dir):
              os.remove(os.path.join(output_dir, fn))
    else:
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    prompts = [
        df_prompt['prompt'][row['from_name']],
        df_prompt['prompt'][row['to_name']]
        ]


    seeds = [row['from_seed'], row['to_seed']]

    duration = row['duration']
    fps = num_interpolation_steps/duration


    latent_width = settings['res_width'] // 8
    latent_height = settings['res_height'] // 8

    from_latent = generate_latent(generator, seeds[0], pipe, latent_height, latent_width)
    to_latent = generate_latent(generator, seeds[1], pipe, latent_height, latent_width)

    from_text_embed = get_text_embed(prompts[0], pipe)
    to_text_embed = get_text_embed(prompts[1], pipe)


    print("Transition {} out of {}".format(i_row, len(df_transitions)))
    print(output_name)
    for i, t in enumerate(tqdm(T)):

        embeds = torch.lerp(from_text_embed, to_text_embed, t)
        # latents = torch.lerp(from_latent, to_latent, t)
        latents = slerp(float(t), from_latent, to_latent)

        pipe_kwargs = gen_pipe_kwargs_transition(t, df_prompt, row['from_name'], row['to_name'], pipe_name, song_name)
        settings['pipe_kwargs'].update(pipe_kwargs)

        with torch.autocast('cuda'):
          images = pipe(
              prompt_embeds=embeds,
              latents = latents,
              **settings['pipe_kwargs']
          )

        clear_output(wait=True)

        output_image = images.images[0]

        output_number_string = str(i).zfill(6)
        output_image.save(os.path.join(output_dir, "frame{}.png".format(output_number_string)))
