#%%
import os
from os.path import join as pjoin
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from aa_utils.local import transition_fn_from_transition_row, clip_names_from_transition_row, image_names_from_transition
from aa_utils.story import read_scene_dict, downselect_to_scene_sequence, gen_path_edges_short, construct_input_image_folder_paths, check_input_image_folders_exist, generate_text_for_ffmpeg, generate_output_video
# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='scene_sequence_kv3', dest='scene_sequence')
parser.add_argument("-n", default=0, type=int, dest='N_repeats')
parser.add_argument('-o', default='story_long.mov', dest='output_filename')
parser.add_argument('-f', default=10, type=int, dest='fps')
# args = parser.parse_args()
args = parser.parse_args("") # Needed for jupyter notebook

N_repeats = args.N_repeats 


from dotenv import load_dotenv; load_dotenv()
gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

G = nx.read_gexf(pjoin(gdrive_basedir, args.song, 'story', 'graph_existing_transitions.gexf'))

# only keep edges with the attribute exists=True
edges_to_keep = [(u,v) for u,v,d in G.edges(data=True) if d['exists']]
G = G.edge_subgraph(edges_to_keep)

# keep largest graph 
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc)


#%%

scene_dict, file_to_scene_dict = read_scene_dict(gdrive_basedir, args.song)

for node in list(G.nodes):
    if node in file_to_scene_dict:
        G.nodes[node]['scene'] = file_to_scene_dict[node]
    else:
        # G.nodes[node]['scene'] = 'missing'
        print("No scene for node: {}, dropping".format(node))
        # Drop this node from the graph
        G.remove_node(node)
        #

# %%
fp_scene_sequence = os.path.join(gdrive_basedir, args.song, 'prompt_data', '{}.csv'.format(args.scene_sequence))
scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)['scene'].values.tolist()

# scene_sequence = scene_sequence[0:5]

#%%

G_sequence = downselect_to_scene_sequence(G, scene_sequence)


#%%

from aa_utils.plot import plot_scene_sequence

plot_scene_sequence(G, scene_sequence, scene_dict)

#%%

G_sel = G_sequence

N_repeats = 1

first_scene = scene_sequence[0]
last_scene = scene_sequence[-1]
# pick a start_node that is a random node in the first scene

start_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == first_scene]
end_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == last_scene]

# node must have a path to any of end_nodes
valid_start_nodes = []
for node in start_nodes:
    if any([nx.has_path(G_sel, node, end_node) for end_node in end_nodes]):
        valid_start_nodes.append(node)

path = [np.random.choice(valid_start_nodes)]

for i, scene in enumerate(scene_sequence):

    # make a subgraph of the nodes in the scene

    scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene]

    scene_graph = G_sel.subgraph(scene_nodes)

    # make a random path of length N_repeats edges starting at start_node



    #TODO: make sure that the path is not a loop
    for j in range(N_repeats):
        scene_neighbors = list(scene_graph.neighbors(path[-1]))

        if len(scene_neighbors) == 0:
            print("No neighbors for node: {}, skipping".format(path[-1]))
            break

        path.append(np.random.choice(scene_neighbors))


    if i < len(scene_sequence) - 1:
        # find a path to a node in the next scene

        next_scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene_sequence[i+1]]

        both_scene_graph = G_sel.subgraph([*scene_nodes, *next_scene_nodes])

        # find the shortest path from the last node in the path to a node in the next scene

        valid_next_nodes = [node for node in next_scene_nodes if nx.has_path(both_scene_graph, path[-1], node)]

        # Make a graph of all nodes in scenes that are beyond the next scene

        remaining_scenes = scene_sequence[i+1:]

        remaining_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] in remaining_scenes]

        remaining_graph = G_sel.subgraph(remaining_nodes)

        # make sure the next node is connected to any of end_nodes within the remaining graph, i.e. we do not have to go back through the current scene

        valid_next_nodes = [node for node in valid_next_nodes if any([nx.has_path(remaining_graph, node, end_node) for end_node in end_nodes])]

        path_to_next_scene = nx.shortest_path(both_scene_graph, path[-1], np.random.choice(valid_next_nodes))

        # add this path to the path

        path.extend(path_to_next_scene[1:])


#%%


path_edges = list(zip(path,path[1:]))

plot_scene_sequence(G_sel, scene_sequence, scene_dict, path=path_edges)

plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'storygraph_long.png'))

#%%


#%%


df_transitions = pd.DataFrame(path_edges, columns=['c1','c2'])

# TODO: this can't be obtained from the graph?
dir_transitions = os.path.join(gdrive_basedir, args.song, 'transition_images')

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]



forward_c_pairs = trans_list
song_basedir = os.path.join(gdrive_basedir, args.song)

df_transitions = construct_input_image_folder_paths(df_transitions, song_basedir, forward_c_pairs)

check_input_image_folders_exist(df_transitions)

out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.mkdir(out_dir)

df_transitions.to_csv(os.path.join(out_dir, 'trans_sequence.csv'))


#%%

out_txt = generate_text_for_ffmpeg(df_transitions, fps=args.fps)

# use out_text to make a text file that can be used by ffmpeg to make a movie

with open(os.path.join(out_dir, 'videos.txt'), 'w') as f:
    f.write(out_txt)

generate_output_video(args.fps, out_dir, args.output_filename)
