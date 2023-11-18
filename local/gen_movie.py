
import os
import pandas as pd
import argparse

from aa_utils.story import generate_text_for_ffmpeg, generate_output_video

from dotenv import load_dotenv; load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('-i', default='trans_sequence_long', dest='input_transitions_filename')
parser.add_argument('-o', default='story_long', dest='output_filename')
parser.add_argument('--fps', default=10, type=int, dest='fps')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')


song_basedir = os.path.join(gdrive_basedir, args.song)
story_dir = os.path.join(song_basedir, 'story')

fp_transitions = os.path.join(story_dir, '{}.csv'.format(args.input_transitions_filename))

df_transitions = pd.read_csv(fp_transitions, index_col=0)

out_dir = os.path.join(story_dir, 'sections')


for section, df in df_transitions.groupby('section'):

    print("Generating video for section: {}".format(section))

    out_txt = generate_text_for_ffmpeg(df, fps=args.fps)

    # use out_text to make a text file that can be used by ffmpeg to make a movie

    with open(os.path.join(out_dir, 'videos.txt'), 'w') as f:
        f.write(out_txt)

    generate_output_video(args.fps, out_dir, "{}_{}.mov".format(args.output_filename, section ))

