

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

fp = os.path.join(gdrive_basedir, song, 'intrascene_transitions.csv')
df_trans_intrascene = pd.read_csv(fp, index_col=0).dropna(how='all')
df_trans_intrascene[['c1','c2']] = df_trans_intrascene.apply(clip_names_from_transition_row, axis=1, result_type='expand')

fp = os.path.join(gdrive_basedir, song, 'interscene_transitions.csv')
df_trans_interscene = pd.read_csv(fp, index_col=0).dropna(how='all')
df_trans_interscene[['c1','c2']] = df_trans_interscene.apply(clip_names_from_transition_row, axis=1, result_type='expand')

# Determine clip names for each transition

#%%

# we will develop transitions to the scenese in the following order
df_sequence = pd.read_csv(os.path.join(gdrive_basedir, song, 'scene_sequence.csv'), index_col=0)


df_sequence

# %%

from utils import gendf_trans_sequence

dfs_scenes = []

start_clip = None

for idx, row in df_sequence.iterrows():
    scene=row['scene']
    duration=row['duration']
    scene_folder = os.path.join(input_basedir, scene)

    if not os.path.exists(scene_folder):
        print("missing scene {}".format(scene_folder))
        continue


    # Make series of 


    df_trans_scene = df_trans_intrascene.where(df_trans_intrascene['scene'] == scene).dropna(how='all')

    df_trans_sequence = gendf_trans_sequence(df_trans_scene, num_output_rows=duration, start_clip=start_clip)
    
    # lookup the clips for each transition, and whether they should the reversed clip
    forward_c_pairs = [tuple(c_pair) for c_pair in df_trans_scene[['c1','c2']].values]

    df_trans_sequence['reversed'] = [tuple(c_pair) not in forward_c_pairs  for c_pair in df_trans_sequence[['c1','c2']].values]
    c1s = df_trans_sequence.apply(lambda x: x['c1'] if not x['reversed'] else x['c2'], axis=1)
    c2s = df_trans_sequence.apply(lambda x: x['c2'] if not x['reversed'] else x['c1'], axis=1)
    df_trans_sequence['c1'] = c1s
    df_trans_sequence['c2'] = c2s

    dfs_scenes.append(df_trans_sequence)


    if idx == df_sequence.index[-1]:
        break

    # Find a valid interscene transition from this scene
    out_clip = df_trans_sequence.iloc[-1]['c2']

    trans_from_this_scene = df_trans_interscene[df_trans_interscene['scene_from'] == scene]
    valid_trans_out = trans_from_this_scene[trans_from_this_scene['c1'] == out_clip].iloc[0]

    start_clip = valid_trans_out['c2']
    
    dfs_scenes.append(valid_trans_out.to_frame().T)

    

    #Now generate the interscene transition


#%%

df_trans_sequence = pd.concat(dfs_scenes).reset_index(drop=True)
df_trans_sequence = df_trans_sequence[['c1','c2', 'reversed']]
df_trans_sequence

#%%

df_trans_sequence["input_image_folder"]=df_trans_sequence['c1']+ ' to ' +df_trans_sequence['c2']
# df_trans_sequence['input_movie_folder'] = ['transitions_rev' if reverse else 'transitions' for reverse in df_trans_sequence['reversed']]
song_basedir = os.path.join(gdrive_basedir, song)

df_trans_sequence['input_image_folder'] = song_basedir + '\\' + 'transition_images' + '\\' +  df_trans_sequence['input_image_folder']

df_exist = df_trans_sequence['input_image_folder'].apply(os.path.exists)
df_not_exist =df_trans_sequence.where(df_exist == False).dropna(how='all')

#TODO: move file exist checks to original for loop, such that it can keep trying to make a valid superscene with partial transitions. 
if len(df_not_exist):
    print("Files not existing:  ")
    print(df_not_exist['input_image_folder'].values)
    raise ValueError()

#%%

out_txt = ''
fps = 10
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

out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.mkdir(out_dir)

with open('videos.txt', 'w') as f:
    f.write(out_txt)

import shutil

shutil.move('videos.txt', os.path.join(out_dir, 'videos_story.txt'))

os.chdir(out_dir)
os.system('ffmpeg -f concat -safe 0 -i videos_story.txt -c mjpeg -r {} output_test.mov'.format(fps))
# %%
