#%%
import os
from os.path import join as pjoin
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import image_names_from_transition, build_graph_scenes, check_existing_transitions, gen_scene_dicts
from aa_utils.story import downselect_to_scene_sequence, gen_path_edges_short, generate_text_for_ffmpeg, generate_output_video
from aa_utils.plot import plot_path_labels, plot_scene_sequence

from dotenv import load_dotenv; load_dotenv()
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='scene_sequence_3_la', dest='scene_sequence')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook

gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(args.song))

#%%
fp_scene_sequence = os.path.join(gdrive_basedir, args.song, 'prompt_data', '{}.csv'.format(args.scene_sequence))
df_scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)

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

#%%

path_edges = gen_path_edges_short(G_sequence, scene_sequence_list)

#%%

# TODO: improve
# plot_path_labels(G_sequence, path_edges)

plot_scene_sequence(G_sequence, scene_sequence_list, scene_dict, path_edges=path_edges)

plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'storygraph_short.png'))
# %%
from aa_utils.local import gen_df_transitions, check_input_image_folders_exist

song_basedir = os.path.join(gdrive_basedir, args.song)
out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.makedirs(out_dir)

scene_section_map = df_scene_sequence.set_index('scene')['section']
section_list = [scene_section_map[G.nodes[e[0]]['scene']] for e in path_edges]
section_list.append(scene_section_map[G.nodes[path_edges[-1][1]]['scene']])

df_transitions = gen_df_transitions(G_sequence,path_edges,section_list,song_basedir)

check_input_image_folders_exist(df_transitions)

df_transitions.to_csv(os.path.join(out_dir, 'trans_sequence_short.csv'))

