

#%%
import os
import pandas as pd
import argparse

LIMIT_PROMPTS = 3# Artificually limit number of images, set to None to not use. 

USE_DEFAULT_ARGS = False
if USE_DEFAULT_ARGS:
    song = 'spacetrain'
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    song = args.song

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.
gdrive_basedir = os.getenv('base_dir')

import re
from utils import gendf_imagefn_info

df_sequence = pd.read_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'scene_sequence.csv'), index_col=0)
# We assume the scene list csv has the scenes in order 
ordered_scene_list = df_sequence['scene'].values

dfs = []

for scene in ordered_scene_list:

    scene_basedir = os.path.join(gdrive_basedir,"{}\scenes\{}".format(song, scene))

    scene_dir = os.path.join(scene_basedir)

    if not os.path.isdir(scene_dir):
        raise ValueError("Could not find scene directory: {}".format(scene_dir))

    fns_images = [f for f in os.listdir(scene_dir) if f.endswith('.png')]

    df_imagefn_info = gendf_imagefn_info(fns_images)


    # Generate a sequence of the form
    # sequence = [
    #     (0,1),
    #     (0,2),
    #     (0,3),
    #     (0,4),
    #     (1,2),
    #     (1,3),
    #     (1,4),
    #     (2,3),
    #     (2,4),
    #     (3,4),
    # ]

    if LIMIT_PROMPTS:
        num_prompts = LIMIT_PROMPTS 
    else:
        num_prompts = len(df_imagefn_info)

    sequence = []
    for i in range(num_prompts):
        for j in range(i+1,num_prompts):
            sequence.append((i,j))



    df_transitions = pd.DataFrame(
    index = range(len(sequence)),
    columns = ['from_name', 'from_seed','to_name', 'to_seed', 'compute','duration','scene']

    )

    for i, (start, stop) in enumerate(sequence):
        row_from = df_imagefn_info.loc[start]
        row_to = df_imagefn_info.loc[stop]


        df_transitions.loc[i]['from_name']  = row_from['prompt']
        df_transitions.loc[i]['from_seed']  = row_from['seed']
        df_transitions.loc[i]['to_name']  = row_to['prompt']
        df_transitions.loc[i]['to_seed']  = row_to['seed']



    df_transitions['compute'] = 'y'
    df_transitions['duration'] = 5
    df_transitions['scene'] = scene


    df_transitions['from_seed'] = df_transitions['from_seed'].astype(str)
    df_transitions['to_seed'] = df_transitions['to_seed'].astype(str)

    dfs.append(df_transitions)

df_all = pd.concat(dfs).reset_index(drop=True)

fp_out = os.path.join(gdrive_basedir, song, 'prompt_data', 'intrascene_transitions.csv')
print("writing transitions csv to {}".format(fp_out))
df_all.to_csv(fp_out)


# %%
