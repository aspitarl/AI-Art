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
import shutil
from tqdm import tqdm

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
shutil.copy(json_fp, os.path.join(output_basedir, 'tgen_settings.json'))

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


# %%
device = "cuda"
generator = torch.Generator(device=device)

prompt_sel = ['geo1', 'geo_simple1', 'portal2', 'geom1']
# prompt_sel = ['ridin2', 'elatrain2', 'mushrooms2', 'flyin2']
df_prompt = df_prompt.loc[prompt_sel]

skip_existing = True


combos = {

    # 'num_inference_steps': [5,10,15,20],
    'controlnet_conditioning_scale': [1.5,1.75,2.0,2.25],
    # 'control_guidance_start': [0,0.1,0.2],
    # 'control_guidance_end': [0.6,0.7,0.8,0.9,1.0],
    # 'control_guidance_width': [0.1,0.2,0.3,0.4],

    # 'mask_name': ['window_net', 'window_net_blur', 'window_net_blur_less'],
    'mask_name' : ['circle0', 'circle_oct', 'circle_hex1', 'circle_hex2', 'circle_hex3']
}

with open(os.path.join(output_basedir, 'combos.json'), 'w') as f:
    json.dump(combos, f, indent=4)

image_output_dir = os.path.join(output_basedir, 'images')
if not os.path.exists(image_output_dir): os.makedirs(image_output_dir)

for name, row in df_prompt.iterrows():

    seeds = row['seeds'].split(seed_delimiter)
    seeds = [s.strip() for s in seeds]
    seeds = [int(s) for s in seeds]

    seeds = [seeds[0]] # Keep only the first seed 

    prompt = row['prompt']
    guidance_scale = float(row['guidance_scale'])


    setting_vals = [combos[key] for key in combos.keys()]
    # Use itertools.product to generate all combinations of seeds and cnet_vals
    # for seed, cnet_val, mask_name in itertools.product(seeds, *setting_vals):
    for vals in tqdm(itertools.product(seeds, *setting_vals)):
        seed = vals[0]
        combo_keys = list(combos.keys())

        output_fn = "{}_{}".format(name, seed)
        #TODO: hack to get length, how to get order or handle better
        if 'control_guidance_width' in combos:
            start = vals[1] if 'control_guidance_start' in combos else settings['pipe_kwargs']['control_guidance_start']
            c_end = start + vals[2]
            settings['pipe_kwargs']['control_guidance_end'] = c_end
            
            output_fn += "_gw{}".format(str(vals[2]).replace('.', 'p'))


        for i, key in enumerate(combo_keys):
            val_str = str(vals[i+1])
            if key == 'mask_name':
                output_fn += "_{}".format(val_str.replace('_', ''))
                settings['pipe_kwargs']['image'] = Image.open(os.path.join(os.getenv('REPO_DIR'), 'cloud', 'masks', vals[i+1] + '.png'))
            elif key == 'controlnet_conditioning_scale':
                output_fn += "_{}".format(val_str.replace('.', 'p'))
                settings['pipe_kwargs'][key] = vals[i+1]
            elif key == 'control_guidance_width':
                continue
            else:
                output_fn += "_{}".format(val_str)
                settings['pipe_kwargs'][key] = vals[i+1]

        output_fn += ".png"

        if os.path.exists(os.path.join(image_output_dir, output_fn)):
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

        output_image.save(os.path.join(image_output_dir, output_fn))