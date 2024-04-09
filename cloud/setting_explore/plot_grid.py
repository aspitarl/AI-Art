"""
Folder contains a list of files with filename file_basename_param1_pram2.png

We want to plot the images in a grid with param1 on the x-axis and param2 on the y-axis
"""

import matplotlib.pyplot as plt
import os
import re
import argparse 

parser = argparse.ArgumentParser(description='Generate transitions between prompts')
parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--input_dir', '-i', type=str, help='The input directory for the images', default='test1')
args = parser.parse_args()
song_name = args.song_name

# filename_base = first group, param1 = second group, param2 = third group
regex = re.compile(r'(.*?)_([^_]*)_([^_]*).png')

# input_dir = 'output/escape_cnet/setting_explore'
input_dir = os.path.join('output', song_name, args.input_dir)

output_dir = os.path.join('output', song_name, args.input_dir + '_grid')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fns = [f for f in os.listdir(input_dir)]

# get a list of basenames for each file 

file_basename = [regex.match(fn).group(1) for fn in fns]
file_basename = list(set(file_basename))

# make a dict of files associated with each basename

file_dict = {fb: [fn for fn in fns if fn.startswith(fb)] for fb in file_basename}

for file_basename, fns in file_dict.items():
    param1_vals = list(set([regex.match(fn).group(2) for fn in fns]))
    param1_vals = sorted(param1_vals)
    param2_vals = list(set([regex.match(fn).group(3) for fn in fns]))
    param2_vals = sorted(param2_vals)



    # Create a grid of subplots
    # have the lesser of the two be the number of columns

    if len(param1_vals) > len(param2_vals):
        col_vals = param2_vals
        row_vals = param1_vals
        flip=True
    else:
        col_vals = param1_vals
        row_vals = param2_vals
        flip=False

    fig_size = (3*len(col_vals), 3*len(row_vals))

    fig, axs = plt.subplots(len(row_vals), len(col_vals), figsize=(20, 20))

    for i, col in enumerate(col_vals):
        for j, row in enumerate(row_vals):

            # fn = f'{file_basename}_{col_vals[i]}_{row_vals[i]}.png'
            if flip:
                fn = f'{file_basename}_{row}_{col}.png'
            else:
                fn = f'{file_basename}_{col}_{row}.png'

            if fn in fns:
                img = plt.imread(os.path.join(input_dir, fn))
                axs[j, i].imshow(img)
                axs[j, i].axis('off')
                axs[j, i].set_title(f'{col}, {row}')

    plt.tight_layout()



    plt.savefig(os.path.join(output_dir, file_basename + '.png'))

