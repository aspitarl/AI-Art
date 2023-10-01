

#%%
import os
from os.path import join as pjoin
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from local.utils import transition_fn_from_transition_row, clip_names_from_transition_row, image_names_from_transition
# %%

import argparse

USE_DEFAULT_ARGS = True
if USE_DEFAULT_ARGS:
    song = 'emitnew'
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

scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

# build a list with a random element of scene_dict for each key in scene_sequence

scene_sequence = pd.read_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'scene_sequence.csv'), index_col=0)['scene'].values.tolist()

scene_sequence
# Make a mapping from file to folder name for each scene folder in scene dir

regex = re.compile("([\S\s]+_\d\d\d\d)\d+.png")

scene_dict = {}
for scene in scene_sequence:
    scene_dict[scene] = [fn for fn in os.listdir(pjoin(scene_dir, scene)) if fn.endswith('.png')]

    scene_dict[scene] = [regex.match(fn).groups()[0].replace("_","-") for fn in scene_dict[scene]]

scene_dict


#%%

G = nx.Graph()

# add nodes for each image in each scene

for scene in scene_dict:
    G.add_nodes_from(scene_dict[scene], scene=scene)

scene_names = list(scene_dict.keys())

for i in range(len(scene_names) - 1):
    scene_from = scene_names[i]
    scene_to = scene_names[i+1]

    # add edges between all pairs of nodes in the two scenes
    for node_from in scene_dict[scene_from]:
        for node_to in scene_dict[scene_to]:
            G.add_edge(node_from, node_to)

#%%

dir_transitions = os.path.join(gdrive_basedir, song, 'transition_images')

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

trans_list

#%%

list(G.edges())

any([t in trans_list for t in G.edges()])

#%%

# color edges that are in trans_list differently


# nx.draw(G, node_color=colors, edge_color=edge_colors, with_labels=True, node_size=50, labels=nx.get_node_attributes(G,'label'))


# G.add_nodes_from(nodes)
# G.add_edges_from(trans_list)


#%%

plt.figure(figsize=(10,10))

# Make a color map with a different color for each scene based on the scene of each node

# create number for each group to allow use of colormap

from itertools import count
# get unique groups

groups = set(nx.get_node_attributes(G,'scene').values())
mapping = dict(zip(sorted(groups),count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['scene']] for n in nodes]

edge_colors = []

for edge in G.edges():
    if edge in trans_list:
        edge_colors.append('green')
    else:
        edge_colors.append('red')

# drawing nodes and edges separately so we can capture collection for colobar

pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, edge_color= edge_colors, alpha=0.2)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)

plt.colorbar(nc)
plt.axis('off')


