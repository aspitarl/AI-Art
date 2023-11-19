#%%
import os
import pandas as pd
import argparse

from aa_utils.story import generate_text_for_ffmpeg, generate_output_video

from dotenv import load_dotenv; load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('-i', default='trans_sequence_gen', dest='input_transitions_filename')
parser.add_argument('-o', default='story_long', dest='output_filename')
parser.add_argument('--fps', default=10, type=int, dest='fps')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')


song_basedir = os.path.join(gdrive_basedir, args.song)
story_dir = os.path.join(song_basedir, 'story')

fp_transitions = os.path.join(story_dir, '{}.csv'.format(args.input_transitions_filename))

df_transitions = pd.read_csv(fp_transitions, index_col=0)

#%%

# TODO: this is a lot of redundant code from local/story_nx.py

from os.path import join as pjoin
from aa_utils.local import image_names_from_transition, construct_input_image_folder_paths
import re 

dir_transitions = os.path.join(song_basedir, 'transition_images')

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

forward_c_pairs = trans_list

df_temp = df_transitions[['c1', 'c2']]

# truncate the digits at the end of the string to the first 4 digits with regex sub

truncate_digits = 4

df_temp = df_temp.applymap(lambda x: re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:truncate_digits], x))

# trans_c_pairs = df_temp[['c1', 'c2']].apply(tuple, axis=1)

df_transitions[['c1', 'c2']] = df_temp[['c1', 'c2']]

#TODO: just using this to determine file path, but reversed should already be accurate?
df_transitions = construct_input_image_folder_paths(df_transitions, song_basedir, forward_c_pairs)

#%%

out_dir = os.path.join(story_dir, 'sections')

for section, df in df_transitions.groupby('section'):

    print("Generating video for section: {}".format(section))

    out_txt = generate_text_for_ffmpeg(df, fps=args.fps)

    # use out_text to make a text file that can be used by ffmpeg to make a movie

    with open(os.path.join(out_dir, 'videos.txt'), 'w') as f:
        f.write(out_txt)

    generate_output_video(args.fps, out_dir, "{}_{}.mov".format(args.output_filename, section ))

