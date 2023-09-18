

#%%
import os
import re
import numpy as np
import pandas as pd

from utils import transition_fn_from_transition_row, clip_names_from_transition_row
# %%

import argparse

USE_DEFAULT_ARGS = True
if USE_DEFAULT_ARGS:
    song = 'spacetrain_1024'
    num_output_rows = int(10)
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    parser.add_argument("num_output_rows")
    args = parser.parse_args()

    song = args.song
    num_output_rows = int(args.num_output_rows)

from dotenv import load_dotenv; load_dotenv()
gdrive_basedir = os.getenv('base_dir')
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(song))

#%%
# Load list of all transitions

fp = os.path.join(gdrive_basedir, song, 'all_transitions.csv')
df_transitions = pd.read_csv(fp, index_col=0).dropna(how='all')

# Determine clip names for each transition
df_transitions['fn'] = df_transitions.apply(transition_fn_from_transition_row, axis=1)
df_transitions[['c1','c2']] = df_transitions.apply(clip_names_from_transition_row, axis=1, result_type='expand')

#%%

# we will develop transitions to the scenese in the following order

df_sequence = pd.DataFrame(
    {
        'scene': ['s1', 's2','s3'],
        'duration': [3,3,3]
    }
)


df_sequence
# %%

df_trans_intrascene = df_transitions.dropna(subset=['scene'])
df_trans_interscene = df_transitions.dropna(subset=['scene_from'])

from utils import gendf_trans_sequence


dfs_scenes = []

for idx, row in df_sequence.iterrows():
    scene=row['scene']
    duration=row['duration']
    scene_folder = os.path.join(input_basedir, scene)

    if not os.path.exists(scene_folder):
        print("missing scene {}".format(scene_folder))
        continue


    # Make series of 


    df_trans_scene = df_transitions.where(df_transitions['scene'] == scene).dropna(how='all')

    df_trans_sequence = gendf_trans_sequence(df_trans_scene, num_output_rows=duration)
        
    # lookup the clips for each transition, and whether they should the reversed clip
    forward_c_pairs = [tuple(c_pair) for c_pair in df_trans_scene[['c1','c2']].values]

    df_trans_sequence['reversed'] = [tuple(c_pair) not in forward_c_pairs  for c_pair in df_trans_sequence[['c1','c2']].values]
    df_trans_sequence['from_seed'] = df_trans_sequence.apply(lambda x: x['c1'] if not x['reversed'] else x['c2'], axis=1)
    df_trans_sequence['to_seed'] = df_trans_sequence.apply(lambda x: x['c2'] if not x['reversed'] else x['c1'], axis=1)

    dfs_scenes.append(df_trans_sequence)



# %%

dfs_out = []


for i in range(len(dfs_scenes) -1):

    df_scene_from = dfs_scenes[i]
    df_scene_to = dfs_scenes[i+1]

    from_seed_inteseed = df_scene_from.iloc[-1]['to_seed']
    to_seed_interseed = df_scene_to.iloc[0]['from_seed']

    df_interscene = pd.DataFrame([[None, None, False, from_seed_inteseed, to_seed_interseed]], columns = df_scene_from.columns)

    dfs_out.extend([df_scene_from, df_interscene, df_scene_to])


df_out = pd.concat(dfs_out)

df_out

#%%
#TODO: improve this name 

df_trans_sequence = df_out


df_trans_sequence["input_image_folder"]=df_trans_sequence['from_seed']+ ' to ' +df_trans_sequence['to_seed']
# df_trans_sequence['input_movie_folder'] = ['transitions_rev' if reverse else 'transitions' for reverse in df_trans_sequence['reversed']]
song_basedir = os.path.join(gdrive_basedir, song)

df_trans_sequence['input_image_folder'] = song_basedir + '\\' + 'transition_images' + '\\' +  df_trans_sequence['input_image_folder']

df_exist = df_trans_sequence['input_image_folder'].apply(os.path.exists)
df_not_exist =df_trans_sequence.where(df_exist == False).dropna(how='all')

#TODO: move file exist checks to original for loop, such that it can keep trying to make a valid superscene with partial transitions. 
if len(df_not_exist):
    print("Files not existing:  ")
    print(df_not_exist[['input_movie_folder', 'fn']])
    raise ValueError()

#%%

out_txt = ''
fps = 5
image_duration = 1/fps

for idx, row in df_trans_sequence.iterrows():

    folder = row['input_image_folder']

    images = [fn for fn in os.listdir(folder) if fn.endswith('.png')]

    if row['reversed']: images = images[::-1]

    image_fps = [os.path.join(folder, fn) for fn in images]
    image_fps = [fp.replace('\\', '/') for fp in image_fps]
    image_fps = [fp.replace(' ', '\ ') for fp in image_fps]

    for fp in image_fps:
        out_txt += 'file {}\nduration {}\n'.format(fp, image_duration)



#%%
# out_txt = 'file ' + "\nfile ".join(image_list)

with open('videos.txt', 'w') as f:
    f.write(out_txt)

import shutil

shutil.move('videos.txt', os.path.join(song_basedir, 'videos_story.txt'))
# %%
