#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from itertools import count

from aa_utils.local import gen_scene_dicts, gen_path_sequence_fullG, build_graph_scenes, image_names_from_transition, check_existing_transitions
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

from aa_utils.local import load_df_scene_sequence
df_scene_sequence = load_df_scene_sequence(args.scene_sequence, args.song, dir_option=os.getenv('ss_dir_option'))

# remove 'random' from the start column, replacing with nan
df_scene_sequence['start'] = df_scene_sequence['start'].replace('random', np.nan)

scene_sequence_list = df_scene_sequence['scene'].values.tolist()

scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence_list, truncate_digits=None)

#%%

G = build_graph_scenes(scene_dict)

#%%

path_edges = gen_path_sequence_fullG(G, df_scene_sequence)

#%%

# Check that path_edges is one connected path

path_nodes = set([n for e in path_edges for n in e])

G_path = nx.Graph()
G_path.add_nodes_from(path_nodes)
G_path.add_edges_from(path_edges)

nx.is_connected(G_path)

nx.draw(G_path)

#%%
# Having to check existing separately so that output transitions file are not truncated to 4 digits
# TODO: rework seed length to avoid this and truncation in geenral
# TODO: this can't be obtained from the graph?
dir_transitions = os.path.join(gdrive_basedir, args.song, 'transition_images')
if not os.path.exists(dir_transitions): os.makedirs(dir_transitions)
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

# make a mapping from node name to truncated name for each node in the graph

node_to_trunc = {n: n.split('-')[0] + '-' + n.split('-')[1][:4] for n in G.nodes}

G_plot = G.copy()
G_plot = nx.relabel_nodes(G_plot, node_to_trunc)
G_plot = check_existing_transitions(G_plot, trans_list)
path_edges_truncate = [(node_to_trunc[e[0]], node_to_trunc[e[1]]) for e in path_edges]

plot_scene_sequence(G_plot, scene_sequence_list, scene_dict, path_edges=path_edges_truncate)

if not os.path.exists(pjoin(gdrive_basedir, args.song, 'story')): os.makedirs(pjoin(gdrive_basedir, args.song, 'story'))
plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'story_transition_gen.png'))

# %%
from aa_utils.local import gen_df_transitions

song_basedir = os.path.join(gdrive_basedir, args.song)
out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.makedirs(out_dir)

scene_section_map = df_scene_sequence.set_index('scene')['section']
section_list = [scene_section_map[G.nodes[e[0]]['scene']] for e in path_edges]
section_list.append(scene_section_map[G.nodes[path_edges[-1][1]]['scene']])

df_transitions = gen_df_transitions(G,path_edges,section_list,song_basedir)

df_transitions.to_csv(os.path.join(out_dir, 'trans_sequence_gen.csv'))

#%%
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

df_inter = df_inter.sort_values('scene_from', key=lambda x: x.map(scene_sequence_list.index))
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

    df_intra = df_intra.sort_values('scene', key=lambda x: x.map(scene_sequence_list.index))
    df_intra = df_intra.reset_index(drop=True)

    df_intra


fp_out = os.path.join(gdrive_basedir, args.song, 'prompt_data', 'intrascene_transitions.csv')
print("writing transitions csv to {}".format(fp_out))
df_intra.to_csv(fp_out)

