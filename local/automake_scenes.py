#%%


import os
import pandas as pd
import argparse

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.
gdrive_basedir = os.getenv('base_dir')

from aa_utils.local import gendf_imagefn_info

# we will develop transitions to the scenese in the following order

USE_DEFAULT_ARGS = False
if USE_DEFAULT_ARGS:
    song = 'spacetrain_1024'
    # scene = 'tram_alien'
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    song = args.song

allscenes_folder = os.path.join(gdrive_basedir, song, 'scenes')
prompt_images_folder = os.path.join(gdrive_basedir, song, 'prompt_images')
# %%

# make a list of all png files in prompt_images_folder

fns_prompt_images = [f for f in os.listdir(prompt_images_folder) if f.endswith('.png')]

fns_prompt_images
# %%

import re
regex = re.compile("([\S\s]+)_(\d+).png")

# make a dictionary of a list of images, with keys being the first part of the filename (before the underscore)

prompt_dict = {}
for fn in fns_prompt_images:
    prompt, seed = regex.match(fn).groups()
    if prompt not in prompt_dict:
        prompt_dict[prompt] = []
    prompt_dict[prompt].append(fn)

prompt_dict

#%%

# for each prompt, make a folder in the scenes folder, and copy the images into it using shutil

import shutil

for prompt in prompt_dict:
    scene_dir = os.path.join(allscenes_folder, prompt)
    if not os.path.isdir(scene_dir):
        os.mkdir(scene_dir)
    for fn in prompt_dict[prompt]:
        shutil.copyfile(os.path.join(prompt_images_folder, fn), os.path.join(scene_dir, fn))

# %%

# make a csv file with the scene sequence

df_sequence = pd.DataFrame({'scene':list(prompt_dict.keys())})  

df_sequence['duration'] = 3

df_sequence.to_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'scene_sequence_auto.csv'))

