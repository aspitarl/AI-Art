#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from itertools import count

from aa_utils.local import gen_scene_dicts, gen_transitions_path_edges, transition_fn_from_transition_row, clip_names_from_transition_row, image_names_from_transition

from dotenv import load_dotenv; load_dotenv()
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='scene_sequence_kv3', dest='scene_sequence')
parser.add_argument("-n", default=0, type=int, dest='N_repeats')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

scene_dir = pjoin(gdrive_basedir, args.song, 'scenes')
# scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

fp_scene_sequence = os.path.join(gdrive_basedir, args.song, 'prompt_data', '{}.csv'.format(args.scene_sequence))
scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)['scene'].values.tolist()

scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence, truncate_digits=None)

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

path_edges = gen_transitions_path_edges(G, scene_names, args.N_repeats)

#%%

# Check that path_edges is one connected path

path_nodes = set([n for e in path_edges for n in e])

G_path = nx.Graph()
G_path.add_nodes_from(path_nodes)
G_path.add_edges_from(path_edges)

nx.is_connected(G_path)

nx.draw(G_path)

#%%

plt.figure(figsize=(10,10))

# Make a color map with a different color for each scene based on the scene of each node

# create number for each group to allow use of colormap

# get unique groups

groups = scene_sequence
mapping = dict(zip(groups,count()))
nodes = G.nodes()
colors = [mapping[G.nodes[n]['scene']] for n in nodes]

edge_colors = []
alphas = []

for edge in G.edges():
    edge_rev = (edge[1], edge[0])
    if edge in path_edges or edge_rev in path_edges:
        edge_colors.append('green')
        alphas.append(1)
    else:
        edge_colors.append('red')
        alphas.append(0.1)

# drawing nodes and edges separately so we can capture collection for colobar

pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, edge_color= edge_colors, alpha=alphas)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=5, cmap=plt.cm.jet)

plt.colorbar(nc)
plt.axis('off')


plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'story_transition_gen.png'))

# %%

# iterate through path_edges and split into inter and intra scene edges


interscene_edges = []
intrascene_edges = []

for edge in path_edges:
    scene_from = G.nodes[edge[0]]['scene']
    scene_to = G.nodes[edge[1]]['scene']
    if scene_from == scene_to:
        intrascene_edges.append(edge)
    else:
        interscene_edges.append(edge)


intrascene_edges = list(set(intrascene_edges))
interscene_edges = list(set(interscene_edges))

# %%

# Interscene transition file

# split each edge into two list of nodes_from and nodes_to

nodes_from = [e[0] for e in interscene_edges]
nodes_to = [e[1] for e in interscene_edges]
scenes_from = [G.nodes[n]['scene'] for n in nodes_from]
scenes_to = [G.nodes[n]['scene'] for n in nodes_to]

# make a dataframe wit nodes_from and nodes_to as columns

df_inter = pd.DataFrame({'nodes_from':nodes_from, 'nodes_to':nodes_to})

# Split the nodes_from and nodes_to columns with the hyphen in each node name 

df_inter[['from_name', 'from_seed']] = df_inter['nodes_from'].str.split('-', expand=True)
df_inter[['to_name', 'to_seed']] = df_inter['nodes_to'].str.split('-', expand=True)

# drop the nodes_from an nodes_to columns

df_inter = df_inter.drop(columns=['nodes_from', 'nodes_to'])

df_inter['compute'] = 'y'
df_inter['duration'] = 5
df_inter['scene_from'] = scenes_from
df_inter['scene_to'] = scenes_to


df_inter['from_seed'] = df_inter['from_seed'].astype(str)
df_inter['to_seed'] = df_inter['to_seed'].astype(str)

df_inter = df_inter.sort_values('scene_from', key=lambda x: x.map(scene_sequence.index))
df_inter = df_inter.reset_index(drop=True)


fp_out = os.path.join(gdrive_basedir, args.song, 'prompt_data', 'interscene_transitions.csv')
print("writing transitions csv to {}".format(fp_out))
df_inter.to_csv(fp_out)



# %%


# Intrascene transition file

if len(intrascene_edges) == 0:
    df_intra = pd.DataFrame(columns = ['from_name','from_seed','to_name','to_seed', 'compute', 'duration', 'scene'])

else:

    # split each edge into two list of nodes_from and nodes_to

    nodes_from = [e[0] for e in intrascene_edges]
    nodes_to = [e[1] for e in intrascene_edges]
    scenes_from = [G.nodes[n]['scene'] for n in nodes_from]
    scenes_to = [G.nodes[n]['scene'] for n in nodes_to]

    # assert that scenes_from and scenes_to are the same

    assert all([scenes_from[i] == scenes_to[i] for i in range(len(scenes_from))])

    # make a dataframe wit nodes_from and nodes_to as columns

    df_intra = pd.DataFrame({'nodes_from':nodes_from, 'nodes_to':nodes_to})

    # Split the nodes_from and nodes_to columns with the hyphen in each node name

    df_intra[['from_name', 'from_seed']] = df_intra['nodes_from'].str.split('-', expand=True)

    df_intra[['to_name', 'to_seed']] = df_intra['nodes_to'].str.split('-', expand=True)

    # drop the nodes_from an nodes_to columns

    df_intra = df_intra.drop(columns=['nodes_from', 'nodes_to'])

    df_intra['compute'] = 'y'
    df_intra['duration'] = 5
    df_intra['scene'] = scenes_from

    df_intra['from_seed'] = df_intra['from_seed'].astype(str)
    df_intra['to_seed'] = df_intra['to_seed'].astype(str)

    df_intra = df_intra.sort_values('scene', key=lambda x: x.map(scene_sequence.index))
    df_intra = df_intra.reset_index(drop=True)

    df_intra


fp_out = os.path.join(gdrive_basedir, args.song, 'prompt_data', 'intrascene_transitions.csv')
print("writing transitions csv to {}".format(fp_out))
df_intra.to_csv(fp_out)

