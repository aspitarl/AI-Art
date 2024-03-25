#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import gen_scene_dicts, image_names_from_transition, build_graph_scenes, check_existing_transitions
from aa_utils.plot import plot_scene_sequence

from dotenv import load_dotenv; load_dotenv(override=True)
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

scene_dir = pjoin(gdrive_basedir, args.song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

# scene_sequence_name = "scene_sequence" if args.scene_sequence == '' else "scene_sequence_{}".format(args.scene_sequence)
# fp_scene_sequence = os.path.join(os.getenv('repo_dir'), 'song_meta', args.song, '{}.csv'.format(scene_sequence_name))
# scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)['scene'].values.tolist()
from aa_utils.local import load_df_scene_sequence
df_scene_sequence = load_df_scene_sequence(args.scene_sequence, args.song, dir_option=os.getenv('ss_dir_option'))
scene_sequence = df_scene_sequence['scene'].values.tolist()


# Make a mapping from file to folder name for each scene folder in scene dir
# We truncate here as transition folders are truncated to 4 digits...
scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence, truncate_digits=4)

#%%

dir_transitions = os.path.join(gdrive_basedir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

G = build_graph_scenes(scene_dict)
G = check_existing_transitions(G, trans_list)

if not os.path.exists(pjoin(gdrive_basedir, args.song, 'story')): os.makedirs(pjoin(gdrive_basedir, args.song, 'story'))
nx.write_gexf(G, pjoin(gdrive_basedir, args.song, 'story', 'graph_existing_transitions.gexf'))


#%%

plot_scene_sequence(G, scene_sequence, scene_dict)

plt.tight_layout()
plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'graph_existing_transitions.png'))

# %%

# Make a csv of the existing transitions
# TODO: get full seed from each node, not just the first 4 digits

# split each edge into two list of nodes_from and nodes_to

nodes_from = [e[0] for e in trans_list]
nodes_to = [e[1] for e in trans_list]

# make a dataframe wit nodes_from and nodes_to as columns
df_existing = pd.DataFrame({'nodes_from':nodes_from, 'nodes_to':nodes_to})

if len(df_existing):

    df_existing[['from_name', 'from_seed']] = df_existing['nodes_from'].str.split('-', expand=True)
    df_existing[['to_name', 'to_seed']] = df_existing['nodes_to'].str.split('-', expand=True)

    df_existing = df_existing.drop(columns=['nodes_from', 'nodes_to'])

#%%

df_existing.to_csv(os.path.join(gdrive_basedir, args.song, 'prompt_data', 'existing_transitions.csv'))