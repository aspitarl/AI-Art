#%%
import os
import re
import numpy as np
import pandas as pd

from utils import transition_fn_from_transition_row, clip_names_from_transition_row
# %%

import argparse

USE_DEFAULT_ARGS = False
if USE_DEFAULT_ARGS:
    song = 'spacetrain_1024'
    scene = 's1'
    num_output_rows = 5
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    parser.add_argument("scene")
    parser.add_argument("num_output_rows")
    args = parser.parse_args()

    song = args.song
    scene = args.scene
    num_output_rows = int(args.num_output_rows)

from dotenv import load_dotenv; load_dotenv()
media_dir = os.getenv('media_dir')
input_basedir = os.path.join(media_dir, '{}\scenes'.format(song))

scene_folder = os.path.join(input_basedir, scene)

if not os.path.exists(scene_folder):
    raise ValueError("did not find input scene folder")

folder_rev = os.path.join(scene_folder, 'rev')

#%%
# Load list of all transitions

fp = os.path.join(media_dir, song, 'all_transitions.csv')
df_transitions = pd.read_csv(fp, index_col=0).dropna(how='all')

# Determine clip names for each transition
df_transitions['fn'] = df_transitions.apply(transition_fn_from_transition_row, axis=1)
df_transitions[['c1','c2']] = df_transitions.apply(clip_names_from_transition_row, axis=1, result_type='expand')


#This gets rid of all interscene transitions
df_transitions = df_transitions.where(df_transitions['scene'] == scene).dropna(how='all')

df_transitions

# %%

from utils import gendf_trans_sequence

df_trans_sequence = gendf_trans_sequence(df_transitions, num_output_rows)
    
# lookup the clips for each transition, and whether they should the reversed clip
forward_c_pairs = [tuple(c_pair) for c_pair in df_transitions[['c1','c2']].values]

df_trans_sequence['reversed'] = [tuple(c_pair) not in forward_c_pairs  for c_pair in df_trans_sequence[['c1','c2']].values]
df_trans_sequence['from_seed'] = df_trans_sequence.apply(lambda x: x['c1'] if not x['reversed'] else x['c2'], axis=1)
df_trans_sequence['to_seed'] = df_trans_sequence.apply(lambda x: x['c2'] if not x['reversed'] else x['c1'], axis=1)

df_trans_sequence

# %%

# Find the file path associated with each transition, which depends on whether that is reversed

df_trans_sequence["fn"]=df_trans_sequence['from_seed']+ ' to ' +df_trans_sequence['to_seed']+'.mp4'
df_trans_sequence['input_movie_folder'] = ['transitions_rev' if reverse else 'transitions' for reverse in df_trans_sequence['reversed']]
input_basedir = os.path.join(media_dir, song)

df_trans_sequence['fp_out'] = input_basedir + '\\' + df_trans_sequence['input_movie_folder'] + '\\' +  df_trans_sequence['fn']

df_trans_sequence['fp_out'].values

# %%

# Check if any transition files are missing

df_exist = df_trans_sequence['fp_out'].apply(os.path.exists)
df_not_exist =df_trans_sequence.where(df_exist == False).dropna(how='all')

#TODO: move file exist checks to original for loop, such that it can keep trying to make a valid superscene with partial transitions. 
if len(df_not_exist):
    print("Files not existing:  ")
    raise ValueError(df_not_exist[['output_folder', 'fn']])

df_trans_sequence['fp_out']  = df_trans_sequence['fp_out'].str.replace('\\','/', regex=False)
#%%
out_txt = 'file ' + "\nfile ".join(df_trans_sequence['fp_out'].str.replace(' ', '\ '))

with open('videos.txt', 'w') as f:
    f.write(out_txt)

import shutil

shutil.move('videos.txt', os.path.join(scene_folder, 'videos.txt'))
# %%
