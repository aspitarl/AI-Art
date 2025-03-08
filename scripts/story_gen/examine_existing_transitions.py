#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import json

from aa_utils.local import gen_scene_dicts, image_names_from_transition, build_graph_scenes, check_existing_transitions
from aa_utils.plot import plot_scene_sequence
from aa_utils.local import build_graph_scenes, gen_scene_dict_simple
from aa_utils.fileio import load_df_prompt
from aa_utils.fileio import load_df_scene_sequence

from dotenv import load_dotenv; load_dotenv(override=True)
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--setting_name', '-sn', type=str, default='default', nargs='?', help='Name of top-level key in settings json')
parser.add_argument('--ss', default='', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

media_dir = os.getenv('media_dir')
# media_dir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"

#%%
song_name = args.song
song_meta_dir = os.path.join(os.getenv('meta_dir'), song_name)

df_scene_sequence = load_df_scene_sequence("", song_name)
scene_sequence_list = df_scene_sequence['scene'].tolist()

# load json file with song settings
with open(os.path.join(song_meta_dir, 'tgen_settings.json'), 'r') as f:
    settings = json.load(f)[args.setting_name]

seed_delimiter = settings.get('seed_delimiter', ', ')
df_prompt = load_df_prompt(song_meta_dir, seed_delimiter)

scene_to_file_dict, file_to_scene_dict= gen_scene_dict_simple(df_scene_sequence, df_prompt)

# Old method based on files
# scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence, truncate_digits=4)
G = build_graph_scenes(scene_to_file_dict)


#%%

dir_transitions = os.path.join(media_dir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

G = check_existing_transitions(G, trans_list)


# for edge in G.edges():
#     edge_rev = (edge[1], edge[0])
#     if edge in existing_transitions or edge_rev in existing_transitions:
#         G.edges[edge]['exists'] = True
#     else:
#         G.edges[edge]['exists'] = False

# iterate through edges, and indicate if the edge in an interscene or intrascene transition

for edge in G.edges():
    node1 = edge[0]
    node2 = edge[1]

    node1_scene = file_to_scene_dict[node1]
    node2_scene = file_to_scene_dict[node2]

    if node1_scene == node2_scene:
        G.edges[edge]['transition_type'] = 'intra'
    else:
        G.edges[edge]['transition_type'] = 'inter'

if not os.path.exists(pjoin(media_dir, args.song, 'story')): os.makedirs(pjoin(media_dir, args.song, 'story'))
nx.write_gexf(G, pjoin(media_dir, args.song, 'story', 'graph_existing_transitions.gexf'))

#%%
# drop all edges that are not existing transitions

edges_to_drop = [edge for edge in G.edges() if not G.edges[edge]['exists']]
G_only_existing = G.copy()


G_only_existing.remove_edges_from(edges_to_drop)

nx.write_gexf(G_only_existing, pjoin(media_dir, args.song, 'story', 'graph_only_existing_transitions.gexf'))


#%%

plot_scene_sequence(G, scene_sequence_list, scene_to_file_dict)

plt.tight_layout()
plt.savefig(pjoin(media_dir, args.song, 'story', 'graph_existing_transitions.png'))

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

df_existing.to_csv(os.path.join(media_dir, args.song, 'transition_meta', 'existing_transitions.csv'))