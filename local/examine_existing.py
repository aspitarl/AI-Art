

#%%
import os
from os.path import join as pjoin
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from aa_utils.local import transition_fn_from_transition_row, clip_names_from_transition_row, image_names_from_transition
# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='scene_sequence_kv3', dest='scene_sequence')
# args = parser.parse_args()
args = parser.parse_args("") # Needed for jupyter notebook

song = args.song


from dotenv import load_dotenv; load_dotenv()
gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(song))

#%%

scene_dir = pjoin(gdrive_basedir, song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

fp_scene_sequence = os.path.join(gdrive_basedir, args.song, 'prompt_data', '{}.csv'.format(args.scene_sequence))
scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)['scene'].values.tolist()

# Make a mapping from file to folder name for each scene folder in scene dir
from aa_utils.local import gen_scene_dicts
scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence, truncate_digits=4)


#%%

dir_transitions = os.path.join(gdrive_basedir, song, 'transition_images')
if not os.path.exists(dir_transitions): os.makedirs(dir_transitions)

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

trans_list


#%%

G = nx.Graph()

# add nodes for each image in each scene

for scene in scene_dict:
    G.add_nodes_from(scene_dict[scene], scene=scene)

scene_names = list(scene_dict.keys())

for i in range(len(scene_names)):
    scene_from = scene_names[i]

    # add eges between all pairs of nodes in scene_from

    for node_from in scene_dict[scene_from]:
        for node_to in scene_dict[scene_from]:
            if node_from != node_to:
                G.add_edge(node_from, node_to)

    if i < len(scene_names) - 1:
        scene_to = scene_names[i+1]
        # add edges between all pairs of nodes in the two scenes
        for node_from in scene_dict[scene_from]:
            for node_to in scene_dict[scene_to]:
                G.add_edge(node_from, node_to)

for edge in G.edges():
    edge_rev = (edge[1], edge[0])
    if edge in trans_list or edge_rev in trans_list:
        G.edges[edge]['exists'] = True
    else:
        G.edges[edge]['exists'] = False


if not os.path.exists(pjoin(gdrive_basedir, song, 'story')): os.makedirs(pjoin(gdrive_basedir, song, 'story'))
nx.write_gexf(G, pjoin(gdrive_basedir, song, 'story', 'graph_existing_transitions.gexf'))


#%%

from aa_utils.plot import plot_scene_sequence

plot_scene_sequence(G, scene_sequence, scene_dict)

plt.tight_layout()

plt.savefig(pjoin(gdrive_basedir, song, 'story', 'graph_existing_transitions.png'))



# %%

# Make a csv of the existing transitions
# TODO: get full seed from each node, not just the first 4 digits


# split each edge into two list of nodes_from and nodes_to

nodes_from = [e[0] for e in trans_list]
nodes_to = [e[1] for e in trans_list]
# scenes_from = [G.nodes[n]['scene'] for n in nodes_from]
# scenes_to = [G.nodes[n]['scene'] for n in nodes_to]

# make a dataframe wit nodes_from and nodes_to as columns

df_existing = pd.DataFrame({'nodes_from':nodes_from, 'nodes_to':nodes_to})

if len(df_existing):

    df_existing[['from_name', 'from_seed']] = df_existing['nodes_from'].str.split('-', expand=True)
    df_existing[['to_name', 'to_seed']] = df_existing['nodes_to'].str.split('-', expand=True)


    df_existing = df_existing.drop(columns=['nodes_from', 'nodes_to'])

    df_existing

#%%


df_existing.to_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'existing_transitions.csv'))