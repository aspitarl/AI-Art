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
from aa_utils.local import load_df_scene_sequence
from aa_utils.local import build_graph_scenes, gen_path_sequence_fullG, gen_scene_dict_simple
from aa_utils.plot import plot_scene_sequence
from aa_utils.cloud import load_df_prompt, gen_pipe, gen_pipe_kwargs_static

from dotenv import load_dotenv; load_dotenv(override=True)
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_full', nargs='?')
parser.add_argument('--ss', default='', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

media_dir = os.getenv('media_dir')
song_name = args.song

song_meta_dir = os.path.join(os.getenv('meta_dir'), song_name)

df_scene_sequence = load_df_scene_sequence("", song_name)
scene_sequence_list = df_scene_sequence['scene'].tolist()

df_prompt = load_df_prompt(song_meta_dir)

scene_to_file_dict, file_to_scene_dict= gen_scene_dict_simple(df_scene_sequence, df_prompt)

G = build_graph_scenes(scene_to_file_dict)

#%%


dir_transitions = os.path.join(media_dir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

G = build_graph_scenes(scene_to_file_dict)
G = check_existing_transitions(G, trans_list)

# only keep edges with the attribute exists=True
edges_to_keep = [(u,v) for u,v,d in G.edges(data=True) if d['exists']]
G = G.edge_subgraph(edges_to_keep)

# keep largest graph 
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)


#%%

for node in list(G.nodes):
    if node in file_to_scene_dict:
        G.nodes[node]['scene'] = file_to_scene_dict[node]
    else:
        # G.nodes[node]['scene'] = 'missing'
        print("No scene for node: {}, dropping".format(node))
        # Drop this node from the graph
        G.remove_node(node)
        #

#%%

G_sequence = downselect_to_scene_sequence(G, scene_sequence_list)



# %%
import re

df_scene_sequence2 = df_scene_sequence.copy().reset_index(drop=True) #integer index needed

# if the start value is 'random', pick a random start value from the scene. But we need to select only from nodes that are connected to the previous scene
# I think this still could result in a disconnected graph, if we connect to a path in the previous scene that is not connected to the rest of the scene.
nodes_to_choose_from = []

for idx, row in df_scene_sequence2.iterrows():
    if row['start'] == 'random':

        # If the first row, then just pick a random start image from the scene
        if idx == 0:
            scene = row['scene']
            scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == scene]
            df_scene_sequence2['start'].iloc[idx] = np.random.choice(scene_nodes)
            continue

        scene = row['scene']
        prev_scene = df_scene_sequence2['scene'].iloc[idx-1]
        prev_scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == prev_scene]
        scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == scene]

        connected_nodes = [node for node in scene_nodes if any([G_sequence.has_edge(node, prev_node) for prev_node in prev_scene_nodes])]
        nodes_to_choose_from.append(connected_nodes)

        df_scene_sequence2['start'].iloc[idx] = np.random.choice(connected_nodes)

# alternative method, pick a random start value from the scene. This might not be connected to the previous scene
# df_scene_sequence2['start'] = df_scene_sequence2['start'].apply(lambda x: np.random.choice(G_sequence.nodes) if x == 'random' else x)

# if the start value contains spaces, split on spaces and take a random value. This column can contain missing values. 
df_scene_sequence2['start'] = df_scene_sequence2['start'].apply(lambda x: np.random.choice(x.split(',')) if isinstance(x, str) else x)


# if first row start value is NaN, raise error

if pd.isna(df_scene_sequence2['start'].iloc[0]):
    # raise ValueError("First row start value is NaN, need a start image") 
    print("First row start value is NaN, picking a random start image. Warning, if this image is disconnected you will need to spin again.")
    # pick a random start image

    first_scene = df_scene_sequence2['scene'].iloc[0]
    first_scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == first_scene]
    start_node = np.random.choice(first_scene_nodes)
    df_scene_sequence2['start'].iloc[0] = start_node

df_scene_sequence2['start'] = df_scene_sequence2['start'].ffill()

# scene_dict = {scene: [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:truncate_digits], fn) for fn in scene_dict[scene]] for scene in scene_dict}
string_truncate_length = 4
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




df_scene_sequence2

#%%

scene_to_section_dict  = df_scene_sequence2[['scene','section']].set_index('scene')['section'].to_dict()

scene_order_lookup = df_scene_sequence2['scene'].reset_index().set_index('scene')['index'].to_dict()

#%%
path = []
path_section_list = []

for idx, df_path_section in df_scene_sequence2.groupby('path_section'):

    print("Path Section: ", idx)

    path_section_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] in df_path_section['scene'].values]

    if idx == df_scene_sequence2['path_section'].iloc[-1]:

        last_scene = df_path_section['scene'].iloc[-1]
        last_scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == last_scene]
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


    total_duration = df_path_section['duration'].sum() - 1

    start_node = df_path_section['start'].iloc[0]

    # find all simple paths between start and end, that are of total_duraiton in length


    max_duration_add = 20
    print("looking for paths from {} to {}".format(start_node, end_node))
    for j in range(max_duration_add):
        max_duration = total_duration + j
        print("Trying max_duration: ", max_duration)
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

    print("Number of paths: ", len(all_paths))

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

scene_list

#%%

s1_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == 's1']

s1_nodes

#%%
path_edges = list(zip(path,path[1:]))
path_section_list = path_section_list[1:]

plot_scene_sequence(G_sequence, scene_sequence_list, scene_to_file_dict, path_edges=path_edges)

plt.tight_layout()

plt.savefig(pjoin(media_dir, args.song, 'story', 'storygraph_long.png'))
# %%
from aa_utils.local import gen_df_transitions, check_input_image_folders_exist

song_basedir = os.path.join(media_dir, args.song)
out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.makedirs(out_dir)

df_transitions = gen_df_transitions(G_sequence,path_edges,section_list,song_basedir)

# add path section list as second column
df_transitions['path_section'] = path_section_list
first_cols = ['path_section','section']
cols = first_cols + [col for col in df_transitions if col not in first_cols]
df_transitions = df_transitions[cols]

check_input_image_folders_exist(df_transitions)

df_transitions.to_csv(os.path.join(out_dir, 'trans_sequence.csv'))


# %%
df_transitions
# %%
