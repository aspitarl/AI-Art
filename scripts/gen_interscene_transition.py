#%%


import os
import pandas as pd
import argparse

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.
gdrive_basedir = os.getenv('base_dir')

from utils import gendf_imagefn_info

# we will develop transitions to the scenese in the following order
ordered_scene_list = ['s1','s2','s3']

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

dfs = []
for i_scene in range(len(ordered_scene_list) - 1):
    scene_from = ordered_scene_list[i_scene]
    scene_dir_from = os.path.join(allscenes_folder,scene_from )
    scene_to = ordered_scene_list[i_scene+1]
    scene_dir_to = os.path.join(allscenes_folder,scene_to )

    # scene_dir = os.path.join(allscenes_folder, fold)
    # if not os.path.isdir(scene_dir):
    #     continue

    fns_images_from = [f for f in os.listdir(scene_dir_from) if f.endswith('.png')]
    df_from = gendf_imagefn_info(fns_images_from)

    fns_images_to = [f for f in os.listdir(scene_dir_to) if f.endswith('.png')]
    df_to = gendf_imagefn_info(fns_images_to)


    # Make sequenece
    sequence = []
    for i in range(len(df_from)):
        for j in range(len(df_to)):
            sequence.append((i,j))


    # make and fill transitions df info 

    df_transitions = pd.DataFrame(
    index = range(len(sequence)),
    columns = ['from_name', 'from_seed','to_name', 'to_seed', 'compute','duration','scene']

    )

    for i, (start, stop) in enumerate(sequence):
        row_from = df_from.loc[start]
        row_to = df_to.loc[stop]


        df_transitions.loc[i]['from_name']  = row_from['prompt']
        df_transitions.loc[i]['from_seed']  = row_from['seed']
        df_transitions.loc[i]['to_name']  = row_to['prompt']
        df_transitions.loc[i]['to_seed']  = row_to['seed']



    df_transitions['compute'] = 'y'
    df_transitions['duration'] = 5
    df_transitions['scene_from'] = scene_from
    df_transitions['scene_to'] = scene_to


    df_transitions['from_seed'] = df_transitions['from_seed'].astype(str)
    df_transitions['to_seed'] = df_transitions['to_seed'].astype(str)

    dfs.append(df_transitions)        


df_transitions = pd.concat(dfs)

#%%

df_transitions

fp_out = os.path.join(allscenes_folder, 'interscene_transitions.csv')
print("writing interscene transitions csv to {}".format(fp_out))
df_transitions.to_csv(fp_out)
# %%
