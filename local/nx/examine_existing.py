

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

USE_DEFAULT_ARGS = True
if USE_DEFAULT_ARGS:
    song = 'cycle_mask'
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    song = args.song

from dotenv import load_dotenv; load_dotenv()
gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(song))

#%%

scene_dir = pjoin(gdrive_basedir, song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

scene_sequence = pd.read_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'scene_sequence.csv'), index_col=0)['scene'].values.tolist()

# Make a mapping from file to folder name for each scene folder in scene dir

regex = re.compile("([\S\s]+_\d\d\d\d)\d+.png")

scene_dict = {}
for scene in scene_sequence:
    scene_dict[scene] = [fn for fn in os.listdir(pjoin(scene_dir, scene)) if fn.endswith('.png')]

    # scene_dict[scene] = [regex.match(fn).groups()[0].replace("_","-") for fn in scene_dict[scene]]


    scene_dict[scene] = [fn.rsplit('_', 1)[0] + '-' + fn.rsplit('_', 1)[1] for fn in scene_dict[scene]]

    # remove the .png extension from each filename with a regular expression

    scene_dict[scene] = [re.sub(r'\.png$', '', fn) for fn in scene_dict[scene]]

    # truncate the digits after each hypen to 4 digits

    scene_dict[scene] = [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:4], fn) for fn in scene_dict[scene]]





    



scene_dict

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

#%%

plt.figure(figsize=(10,10))

# Make a color map with a different color for each scene based on the scene of each node

# create number for each group to allow use of colormap

from itertools import count
# get unique groups

groups = scene_sequence
mapping = dict(zip(groups,count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['scene']] for n in nodes]

edge_colors = []
alphas = []

for edge in G.edges():
    edge_rev = (edge[1], edge[0])
    if edge in trans_list or edge_rev in trans_list:
        edge_colors.append('green')
        alphas.append(1)
    else:
        edge_colors.append('red')
        alphas.append(0.1)

# drawing nodes and edges separately so we can capture collection for colobar

pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, edge_color= edge_colors, alpha=alphas)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)

plt.colorbar(nc)
plt.axis('off')

# make one label of the scene name positioned near the center of each scene

scene_centers = {}

for scene in scene_dict:
    scene_centers[scene] = np.mean([pos[n] for n in scene_dict[scene]], axis=0)

for scene in scene_centers:
    plt.text(scene_centers[scene][0], scene_centers[scene][1], scene, fontsize=10, horizontalalignment='center', verticalalignment='center')

if not os.path.exists(pjoin(gdrive_basedir, song, 'story')): os.makedirs(pjoin(gdrive_basedir, song, 'story'))

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