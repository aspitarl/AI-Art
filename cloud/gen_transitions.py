# %%
import os
import pandas as pd
import numpy as np
from IPython.display import clear_output
from aa_utils.sd import generate_latent, get_text_embed, slerp
from aa_utils.cloud import load_df_transitions,load_df_prompt, gen_pipe
import torch
import dotenv; dotenv.load_dotenv()
import argparse
import json
from PIL import Image

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
args = parser.parse_args()
song_name = args.song_name

output_basedir = os.path.join(os.getenv('REPO_DIR'), 'cloud','output', song_name, 'transition_images')
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

dir_prompt_data = os.path.join(os.getenv('REPO_DIR'), 'cloud', 'prompt_data', song_name)
song_meta_dir = os.path.join(os.getenv('REPO_DIR'), 'song_meta', song_name)

# load json file with song settings
json_fp = os.path.join(song_meta_dir, 'tgen_settings.json')
with open(json_fp, 'r') as f:
    settings = json.load(f)


df_prompt = load_df_prompt(song_meta_dir)
df_transitions = load_df_transitions(dir_prompt_data)

pipe_name = 'controlnet' if 'controlnet_string' in settings else 'basic'
pipe = gen_pipe(pipe_name, settings)

if 'mask_image' in settings:
    mask_image = Image.open(os.path.join('masks', settings['mask_image']))
    settings['pipe_kwargs']['image'] = mask_image    

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

    guidance_scales = [
        df_prompt['guidance_scale'][row['from_name']],
        df_prompt['guidance_scale'][row['to_name']]
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

    # The tensor steps are len(num_interpolation_steps) + 1
    # latent_steps = make_latent_steps(from_latent, to_latent, num_interpolation_steps)
    # embed_steps = make_latent_steps(from_text_embed, to_text_embed, num_interpolation_steps)
    guidance_steps = np.linspace(guidance_scales[0], guidance_scales[1], num_interpolation_steps + 1)

    print("Transition {} out of {}".format(i_row, len(df_transitions)))
    print(output_name)
    for i, t in enumerate(tqdm(T)):

        embeds = torch.lerp(from_text_embed, to_text_embed, t)
        # latents = torch.lerp(from_latent, to_latent, t)
        latents = slerp(float(t), from_latent, to_latent)

      

        with torch.autocast('cuda'):
          images = pipe(
              prompt_embeds=embeds,
              guidance_scale=guidance_steps[i],
              latents = latents,
              **settings['pipe_kwargs']
          )

        clear_output(wait=True)

        output_image = images.images[0]

        output_number_string = str(i).zfill(6)
        output_image.save(os.path.join(output_dir, "frame{}.png".format(output_number_string)))
