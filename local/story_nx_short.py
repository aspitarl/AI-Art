#%%
import os
from os.path import join as pjoin
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import transition_fn_from_transition_row, clip_names_from_transition_row, image_names_from_transition
from aa_utils.story import read_scene_dict, downselect_to_scene_sequence, gen_path_edges_short, construct_input_image_folder_paths, check_input_image_folders_exist, generate_text_for_ffmpeg, generate_output_video
# %%

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
parser.add_argument('--ss', default='scene_sequence_kv3', dest='scene_sequence')
parser.add_argument('-o', default='story_short.mov', dest='output_filename')
parser.add_argument('-f', default=10, type=int, dest='fps')
args = parser.parse_args()
# args = parser.parse_args("") # Needed for jupyter notebook


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

# scene_sequence = scene_sequence[0:3]

#%%

G_sequence = downselect_to_scene_sequence(G, scene_sequence)

#%%

path_edges = gen_path_edges_short(G_sequence, scene_sequence)


#%%

from aa_utils.plot import plot_path_labels, plot_scene_sequence
# TODO: improve
# plot_path_labels(G_sequence, path_edges)

plot_scene_sequence(G_sequence, scene_sequence, scene_dict, path=path_edges)

plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'story_graph.png'))
# %%

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
