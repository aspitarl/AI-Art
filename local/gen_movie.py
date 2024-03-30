#%%
import os
import pandas as pd
import argparse

from aa_utils.story import generate_text_for_ffmpeg, generate_output_video

from dotenv import load_dotenv; load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('-i', default='', dest='input_transitions_filename')
parser.add_argument('-o', default='base', dest='output_folder')
parser.add_argument('--fps', default=10, type=int, dest='fps')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')

print(gdrive_basedir)


song_basedir = os.path.join(gdrive_basedir, args.song)
story_dir = os.path.join(song_basedir, 'story')

out_dir = os.path.join(song_basedir, 'stories', args.output_folder)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

name_trans_sequence = 'trans_sequence' if args.input_transitions_filename == '' else "trans_sequence_{}".format(args.input_transitions_filename)
fp_transitions = os.path.join(story_dir, '{}.csv'.format(name_trans_sequence))

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


sections_subfolder = 'sections'
if not os.path.exists(pjoin(out_dir, sections_subfolder)):
    os.makedirs(pjoin(out_dir, sections_subfolder))

df_transitions.to_csv(pjoin(out_dir, 'trans_sequence_{}.csv'.format(args.output_folder)))

for section, df in df_transitions.groupby('section'):

    print("Generating video for section: {}".format(section))

    out_txt = generate_text_for_ffmpeg(df, fps=args.fps)

    # use out_text to make a text file that can be used by ffmpeg to make a movie

    with open(os.path.join(out_dir, 'videos.txt'), 'w') as f:
        f.write(out_txt)

    generate_output_video(args.fps, out_dir, "{}/section_{}.mov".format(sections_subfolder, section))


#%%

# concatenate all output videos into one video called sections_combined.mov in the story_dir

# use out_text to make a text file that can be used by ffmpeg to make a movie
import subprocess

with open(os.path.join(out_dir, 'videos.txt'), 'w') as f:


    for section in df_transitions['section'].unique():

        f.write("file '{}/sections/section_{}.mov'\n".format(out_dir, section))

    # generate_output_video(args.fps, story_dir, "sections_combined.mov")

    # concatenate the videos with subprocess

os.chdir(out_dir)

subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'videos.txt', '-y', '-c', 'mjpeg', '-q:v', '3', '-r', str(args.fps), '{}/{}_combined.mov'.format(out_dir, args.output_folder)])

