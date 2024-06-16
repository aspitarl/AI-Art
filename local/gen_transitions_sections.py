"""
Method of generating transitions file.

path sections are defined by entries 'start' column. 

The 'end' column is the next 'start' value, shifted down by one.

the algorithm is as follows:

1. For each path section, find all nodes that are in the path section and first scene of the next path section
2. If it is the last path section, pick a random end node from the last scene
3. Find all simple paths between the start and end nodes, that are of total_duraiton in length
4. If no paths are found, increase the total_duration by 1 and try again
5. If no paths are found, raise an error
6. If paths are found, check that the scenes go in order 's2', 's3', 's4', etc.
7. If no paths are found in the correct sequence use all paths found
8. If multiple paths are found, select one at random
"""


#%%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import image_names_from_transition, build_graph_scenes, check_existing_transitions, gen_scene_dicts
from aa_utils.story import downselect_to_scene_sequence, gen_path_edges_short, generate_text_for_ffmpeg, generate_output_video
from aa_utils.plot import plot_scene_sequence

from dotenv import load_dotenv; load_dotenv(override=True)
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='emit', nargs='?')
parser.add_argument('--ss', default='', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

media_dir = os.getenv('media_dir')
# media_dir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(media_dir, '{}\scenes'.format(args.song))

#%%

from aa_utils.local import load_df_scene_sequence
df_scene_sequence = load_df_scene_sequence(args.scene_sequence, args.song).reset_index(drop=True)

scene_sequence_list = df_scene_sequence['scene'].values.tolist()

scene_dir = pjoin(media_dir, args.song, 'scenes')
scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence_list, truncate_digits=None)

dir_transitions = os.path.join(media_dir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

G = build_graph_scenes(scene_dict)
G = check_existing_transitions(G, trans_list)

#%%

list(G.nodes)

# %%
import re

df_scene_sequence2 = df_scene_sequence.copy()

# if first row start value is NaN, raise error

if pd.isna(df_scene_sequence2['start'].iloc[0]):
    raise ValueError("First row start value is NaN, need a start image") 

df_scene_sequence2['start'] = df_scene_sequence2['start'].ffill()

# scene_dict = {scene: [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:truncate_digits], fn) for fn in scene_dict[scene]] for scene in scene_dict}
string_truncate_length = 400
df_scene_sequence2['start'] = [re.sub(r'_(\d+)$', lambda m: '-' + m.group(1)[:string_truncate_length], fn) for fn in df_scene_sequence2['start']]

# make an ascending index based on start

df_scene_sequence2['path_section'] = df_scene_sequence2.groupby('start', sort=False).ngroup()

# Create a temporary DataFrame where each path_section only has one row
temp_df = df_scene_sequence2.drop_duplicates('path_section')

# Shift the 'start' values in the temporary DataFrame
temp_df['end'] = temp_df['start'].shift(-1)

# Join the temporary DataFrame back to the original one
df_scene_sequence2 = df_scene_sequence2.merge(temp_df[['path_section', 'end']], on='path_section', how='left')

# pick a random end for the last path_section


#%%

list(G.nodes)

#%%

scene_to_section_dict  = df_scene_sequence2[['scene','section']].set_index('scene')['section'].to_dict()

scene_order_lookup = df_scene_sequence2['scene'].reset_index().set_index('scene')['index'].to_dict()

#%%
path = []
path_section_list = []

G_sequence = G.copy()

for idx, df_path_section in df_scene_sequence2.groupby('path_section'):

    print("Path Section: ", idx)

    path_section_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] in df_path_section['scene'].values]

    if idx == df_scene_sequence2['path_section'].iloc[-1]:

        last_scene = df_path_section['scene'].iloc[-1]
        last_scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == last_scene]
        last_scene_nodes = [node for node in last_scene_nodes if node is not df_path_section['start'].iloc[-1]]
        end_node = np.random.choice(last_scene_nodes)
        
        subgraph = G_sequence.subgraph(path_section_nodes)

    else:
        end_node = df_path_section['end'].iloc[0]


        # Method 1: include entire next scene, can cause movie_section to begin with intrascene transition
        # next_scene = df_scene_sequence2['scene'].iloc[df_path_section.index[-1]+1]
        # next_scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == next_scene]

        # Method 2: force path to go to end_node
        next_scene_nodes = [end_node]

        subgraph = G_sequence.subgraph([*path_section_nodes, *next_scene_nodes])


    total_duration = df_path_section['duration'].sum() + len(df_path_section)

    start_node = df_path_section['start'].iloc[0]

    # find all simple paths between start and end, that are of total_duraiton in length


    max_duration_add = 1

    for j in range(max_duration_add):
        max_duration = total_duration + j
        all_paths = list(nx.all_simple_paths(subgraph, start_node, end_node, cutoff=max_duration))

        if len(all_paths) > 0:
            print("Found paths of length: ", max_duration)
            print("nominal duration was: ", total_duration)
            break
    
    if len(all_paths) == 0:
        print("start_node: ", start_node)
        print("end_node: ", end_node)
        print("subgraph nodes: ", sorted(subgraph.nodes))
        raise ValueError("No valid paths found")

    all_paths_order = []

    for subpath_candidate in all_paths:
        scene_list = [file_to_scene_dict[node] for node in subpath_candidate]
        scene_order = [scene_order_lookup[scene] for scene in scene_list]

        # check that scene_list goes in order 's2', 's3', 's4', etc.
        # sorted_scene_list = sorted(scene_list, key=lambda x: float(x[1:]))
        sorted_scene_order = sorted(scene_order)
        if scene_order == sorted_scene_order:
            all_paths_order.append(subpath_candidate)

    if len(all_paths_order) == 0:
        print("warning, could not find a path that goes in order even though paths were found, sticking with unordered paths for selection")
    else:
        all_paths = all_paths_order

    # What is the length of the longest path?

    max_path_length = max([len(path) for path in all_paths])

    # filter out paths that are not of max_path_length in length

    all_paths = [path for path in all_paths if len(path) == max_path_length]

    # selected_path = np.random.choice(all_paths)
    selected_path_idx = np.random.randint(len(all_paths))
    selected_path = all_paths[selected_path_idx]

    if idx > 0:
        selected_path = selected_path[1:]

    path_section_list.extend([idx]*len(selected_path))
    path.extend(selected_path)

scene_list = [file_to_scene_dict[node] for node in path]
section_list = [scene_to_section_dict[scene] for scene in scene_list]


#%%
path_edges = list(zip(path,path[1:]))
path_section_list = path_section_list[1:]

plot_scene_sequence(G_sequence, scene_sequence_list, scene_dict, path_edges=path_edges)

plt.tight_layout()

plt.savefig(pjoin(media_dir, args.song, 'story', 'storygraph_long.png'))
# %%
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
dir_transitions = os.path.join(media_dir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

# make a mapping from node name to truncated name for each node in the graph

node_to_trunc = {n: n.split('-')[0] + '-' + n.split('-')[1][:4] for n in G.nodes}

G_plot = G.copy()
G_plot = nx.relabel_nodes(G_plot, node_to_trunc)
G_plot = check_existing_transitions(G_plot, trans_list)
path_edges_truncate = [(node_to_trunc[e[0]], node_to_trunc[e[1]]) for e in path_edges]

plot_scene_sequence(G_plot, scene_sequence_list, scene_dict, path_edges=path_edges_truncate)

plt.savefig(pjoin(media_dir, args.song, 'story', 'story_transition_gen.png'))

# %%
from aa_utils.local import gen_df_transitions

song_basedir = os.path.join(media_dir, args.song)
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


fp_out = os.path.join(media_dir, args.song, 'transition_meta', 'interscene_transitions.csv')
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


fp_out = os.path.join(media_dir, args.song, 'transition_meta', 'intrascene_transitions.csv')
print("writing transitions csv to {}".format(fp_out))
df_intra.to_csv(fp_out)

