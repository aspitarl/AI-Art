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
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%

from aa_utils.local import load_df_scene_sequence
df_scene_sequence = load_df_scene_sequence(args.scene_sequence, args.song, dir_option=os.getenv('ss_dir_option'))

scene_sequence_list = df_scene_sequence['scene'].values.tolist()

scene_dir = pjoin(gdrive_basedir, args.song, 'scenes')
scene_dict, file_to_scene_dict = gen_scene_dicts(scene_dir, scene_sequence_list, truncate_digits=4)

dir_transitions = os.path.join(gdrive_basedir, args.song, 'transition_images')
trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

G = build_graph_scenes(scene_dict)
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

#TODO: believe this can be handled by using a directed graph

def remove_hanging_nodes(G_sequence, scene_sequence_list):
    last_scene = scene_sequence_list[-1]
    end_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == last_scene]

    # Iterate through the scene sequence, removing all previous scenes from the graph, and remove any hanging nodes that do not have a path to the end node
    all_valid_scene_nodes = []
    remove_nodes = []
    for i, scene in enumerate(scene_sequence_list[:-1]):

        scene_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == scene]

        remaining_scenes = scene_sequence_list[i:]

        remaining_nodes = [node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] in remaining_scenes]

        remaining_graph = G_sequence.subgraph(remaining_nodes)

        valid_scene_nodes = [node for node in scene_nodes if any([nx.has_path(remaining_graph, node, end_node) for end_node in end_nodes])]

        all_valid_scene_nodes.extend(valid_scene_nodes)

        remove_nodes.extend([node for node in scene_nodes if node not in valid_scene_nodes])

    # add the last scene to the list of valid nodes

    all_valid_scene_nodes.extend([node for node in G_sequence.nodes if G_sequence.nodes[node]['scene'] == last_scene])

    G_sequence = G_sequence.subgraph(all_valid_scene_nodes)

    return G_sequence

# perform remove_hanging_nodes on G_sequence until it no longer changes

print("Removing hanging nodes")
for i in range(10):
    print("Iteration: {}".format(i))
    G_sequence_old = G_sequence.copy()
    G_sequence = remove_hanging_nodes(G_sequence, scene_sequence_list)
    if nx.is_isomorphic(G_sequence, G_sequence_old):
        break

#%%
plot_scene_sequence(G_sequence, scene_sequence_list, scene_dict)

#%%

G_sel = G_sequence


first_scene = scene_sequence_list[0]
last_scene = scene_sequence_list[-1]
# pick a start_node that is a random node in the first scene

start_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == first_scene]
end_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == last_scene]

# node must have a path to any of end_nodes
valid_start_nodes = []
for node in start_nodes:
    if any([nx.has_path(G_sel, node, end_node) for end_node in end_nodes]):
        valid_start_nodes.append(node)



path = [np.random.choice(valid_start_nodes)]

first_section = df_scene_sequence['section'].iloc[0]
section_list = [first_section]

# for i, scene in enumerate(scene_sequence_list[:-1]):
for i, (idx, row) in enumerate(df_scene_sequence.iterrows()):

    if i == len(scene_sequence_list)-1:
        break

    scene = row['scene']
    N_repeats = row['duration']

    current_node = path[-1]

    # make a subgraph of the nodes in the scene

    scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene]

    scene_graph = G_sel.subgraph(scene_nodes)

    # find a path to a node in the next scene

    next_scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene_sequence_list[i+1]]

    both_scene_graph = G_sel.subgraph([*scene_nodes, *next_scene_nodes])

    # find the shortest path from the last node in the path to a node in the next scene

    valid_next_nodes = [node for node in next_scene_nodes if nx.has_path(both_scene_graph, current_node, node)]

    # Make a graph of all nodes in scenes that are beyond the next scene

    if N_repeats == 0:
        path_to_next_scene = nx.shortest_path(both_scene_graph, current_node, np.random.choice(valid_next_nodes))
    else:
        # Make a list of nodes that have an edge with one of the valid_next_nodes
        valid_current_scene_ending_nodes = [node for node in scene_nodes if any([both_scene_graph.has_edge(node, next_node) for next_node in valid_next_nodes])]
        # remove current_node from this list
        if current_node in valid_current_scene_ending_nodes:
            valid_current_scene_ending_nodes.remove(current_node)

        all_simple_paths = [] 
        for valid_end_node in valid_current_scene_ending_nodes:
            simple_paths = nx.all_simple_paths(scene_graph, current_node, valid_end_node, cutoff=N_repeats)
            simple_paths = list(simple_paths)
            if len(simple_paths) > 0:
                all_simple_paths.extend(simple_paths)

        if len(all_simple_paths) == 0:
            print("No simple paths found from node: {}, reverting to shortest path".format(current_node))
            path_to_next_scene = nx.shortest_path(both_scene_graph, current_node, np.random.choice(valid_next_nodes))

        else:
            # i_max = np.argmax([len(path) for path in all_simple_paths])
            # path_to_next_scene = all_simple_paths[i_max]

            # pick a random path from all_simple_paths
            idx_random = np.random.randint(len(all_simple_paths))
            path_to_next_scene = all_simple_paths[idx_random]

            # pick the longest path from all_simple_paths
            # path_to_next_scene = max(all_simple_paths, key=len)

            # print("found simple path: {}".format(path_to_next_scene ))

            edges_from_path = list(both_scene_graph.edges(path_to_next_scene[-1]))

            # find the edge that connects to one of the valid_next_nodes

            for edge in edges_from_path:
                if edge[1] in valid_next_nodes:
                    end_node = edge[1]
                    break

            # assert that an edge exists between path_to_next_scene[-1] and end_node
            assert both_scene_graph.has_edge(path_to_next_scene[-1], end_node)

            # add this edge to the path
            path_to_next_scene.append(end_node)

    # add this path to the path

    path.extend(path_to_next_scene[1:])
    section_list.extend([row['section']] * len(path_to_next_scene[1:]))

# TODO: Add intrascene for the last scene if intrascene edges exist
#     scene_neighbors = list(scene_graph.neighbors(path[-1]))

#%%

path_edges = list(zip(path,path[1:]))

plot_scene_sequence(G_sel, scene_sequence_list, scene_dict, path_edges=path_edges)

plt.tight_layout()

plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'storygraph_long.png'))


#%%

from aa_utils.local import gen_df_transitions, check_input_image_folders_exist

song_basedir = os.path.join(gdrive_basedir, args.song)
out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.makedirs(out_dir)

df_transitions = gen_df_transitions(G_sel,path_edges,section_list,song_basedir)

check_input_image_folders_exist(df_transitions)

df_transitions.to_csv(os.path.join(out_dir, 'trans_sequence.csv'))

