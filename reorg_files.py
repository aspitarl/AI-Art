#%%
import os
import pandas as pd
#%%

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("song")
args = parser.parse_args()

song = args.song

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.
gdrive_basedir = os.getenv('base_dir')

fn = 'input_data.xlsx'
fp = os.path.join(gdrive_basedir, fn)
df = pd.read_excel(fp, sheet_name='transitions_{}'.format(song))
# df = df.dropna(subset=['start'])

df
# %%

movie_dir = os.path.join(gdrive_basedir, "{}\transitions".format(song))
output_basedir = os.path.join(gdrive_basedir, "{}\scenes".format(song))

#%%


for scene_dir in set(df['scene']):
    output_dir = os.path.join(output_basedir, scene_dir)
    if not os.path.exists(output_dir): os.mkdir(output_dir)

# %%

import shutil

max_seed_characters = 4 # Take the first few numbers of the seed for the name

#TODO: This is copied from collab notebook, make into a function

# output_names = []
for i, row in df.iterrows():

    output_name = "{}-{} to {}-{}.mp4".format(
        row['from_name'],
        str(row['from_seed'])[:max_seed_characters],
        row['to_name'],
        str(row['to_seed'])[:max_seed_characters]
        )
    
    scene_dir = row['scene']

    output_fp = os.path.join(output_basedir, scene_dir, output_name)

    input_fp = os.path.join(movie_dir, output_name)
    
    if os.path.exists(input_fp):
        shutil.move(input_fp, output_fp)



  
#   output_names.append(output_name)


df['fp'] = output_name
#%%





