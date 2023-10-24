

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

parser = argparse.ArgumentParser()
parser.add_argument("song", default='cycle_mask_test', nargs='?')
args = parser.parse_args("")


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

scene_dict = pd.read_csv(os.path.join(gdrive_basedir, args.song, 'prompt_data', 'scene_dict.csv'), index_col=0).to_dict()['0']

# convert the values from strings to lists
scene_dict = {k: v.split(',') for k,v in scene_dict.items()}

# remove single quotes and list brackets from each element in the list
scene_dict = {k: [re.sub(r"['\[\]]", '', fn).strip() for fn in v] for k,v in scene_dict.items()}

# truncate the digits after each hyphen to 4 digits
scene_dict = {scene: [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:4], fn) for fn in scene_dict[scene]] for scene in scene_dict}

scene_dict

scene_list = scene_dict.keys()

# invert scene_dict to make a mapping from file to folder name

file_to_scene_dict = {}
for scene in scene_list:

    for fn in scene_dict[scene]:
        file_to_scene_dict[fn] = scene


file_to_scene_dict

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

nx.draw(G)

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

# drawing nodes and edges separately so we can capture collection for colobar
pos = nx.spring_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
plt.colorbar(nc)
plt.axis('off')
# plt.show()
plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'story_graph_2.png'))
# %%

scene_sequence = pd.read_csv(os.path.join(gdrive_basedir, args.song, 'prompt_data', 'scene_sequence.csv'), index_col=0)['scene'].values.tolist()
# scene_sequence = scene_sequence[0:3]

scene_sequence

#%%

# remove all edges from the graph that do not connect nodes in the scene_sequence
# make a list of all edges in the graph

all_edges = list(G.edges)

# only keep edges that connect nodes with pairs of scenes that are adjacent in the scene_sequence
adjacent_edges = [edge for edge in all_edges if G.nodes[edge[0]]['scene'] in scene_sequence and G.nodes[edge[1]]['scene'] in scene_sequence and abs(scene_sequence.index(G.nodes[edge[0]]['scene']) - scene_sequence.index(G.nodes[edge[1]]['scene'])) == 1]

# also include edges that connect nodes in the same scene
adjacent_edges.extend([edge for edge in all_edges if G.nodes[edge[0]]['scene'] == G.nodes[edge[1]]['scene']])

# downselect the graph to only include these edges and their nodes
G_sequence = G.edge_subgraph(adjacent_edges)


#%%

G_sel = G_sequence

first_scene = scene_sequence[0]
last_scene = scene_sequence[-1]

# find the most connected node in the first scene

def most_connected_node(G_sel, scene):
    scene_nodes = [n for n in G_sel.nodes() if G_sel.nodes[n]['scene'] == scene]
    scene_G_sel = G_sel.subgraph(scene_nodes)

    # get the node with the highest degree
    node = max(scene_G_sel.degree, key=lambda x: x[1])[0]

    return node

first_node = most_connected_node(G_sel, first_scene)
last_node = most_connected_node(G_sel, last_scene)

path = nx.shortest_path(G_sel, first_node, last_node)

path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]


path_edges

#%%

last_node = path[-1]

# go to a random node in the last scene that is not last_node

last_scene_nodes = [n for n in G_sel.nodes() if G_sel.nodes[n]['scene'] == last_scene]

# last_scene_nodes = [n for n in last_scene_nodes if n != last_node]

# last_node = np.random.choice(last_scene_nodes)

# final_path = nx.shortest_path(G, first_node, last_node)

# final_path_edges = [(final_path[i], final_path[i+1]) for i in range(len(final_path)-1)]

# path_edges += final_path_edges

last_scene_nodes

# %%
plt.figure(figsize=(6,6))

# Add a label to each node in the path

path = [edge[0] for edge in path_edges]

for i, node in enumerate(path):
    G_sel.nodes[node]['label'] = i


color_map = []

for node in G_sel:
    if node in path:

        color_map.append('green')
    else:
        color_map.append('red')

nx.draw(G_sel, node_color=color_map, with_labels=True, node_size=50, labels=nx.get_node_attributes(G_sel,'label'))

plt.savefig(pjoin(gdrive_basedir, args.song, 'story', 'story_graph.png'))

# %%

df_transitions = pd.DataFrame(path_edges, columns=['c1','c2'])

# TODO: this can't be obtained from the graph?
dir_transitions = os.path.join(gdrive_basedir, args.song, 'transition_images')

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]


forward_c_pairs = trans_list
df_transitions['reversed'] = [tuple(c_pair) not in forward_c_pairs  for c_pair in df_transitions[['c1','c2']].values]

df_transitions

# %%

df_trans_sequence = df_transitions

df_trans_sequence['input_image_folder'] = df_trans_sequence.apply(
    lambda x: x['c1']+ ' to ' + x['c2'] if not x['reversed'] else 
    x['c2']+ ' to ' + x['c1'],
    axis=1
    )

# df_trans_sequence['input_movie_folder'] = ['transitions_rev' if reverse else 'transitions' for reverse in df_trans_sequence['reversed']]
song_basedir = os.path.join(gdrive_basedir, args.song)

df_trans_sequence['input_image_folder'] = song_basedir + '\\' + 'transition_images' + '\\' +  df_trans_sequence['input_image_folder']

df_exist = df_trans_sequence['input_image_folder'].apply(os.path.exists)
df_not_exist =df_trans_sequence.where(df_exist == False).dropna(how='all')

#TODO: move file exist checks to original for loop, such that it can keep trying to make a valid superscene with partial transitions. 
if len(df_not_exist):
    print("Files not existing:  {}".format(df_not_exist))
    print(df_not_exist['input_image_folder'].values)
    raise ValueError()


#%%

out_txt = ''
fps = 5
image_duration = 1/fps

for idx, row in df_trans_sequence.iterrows():

    folder = row['input_image_folder']

    images = [fn for fn in os.listdir(folder) if fn.endswith('.png')]
    images = sorted(images)

    if row['reversed']: images = images[::-1]

    # remove the last element of the list

    images = images[:-1]

    image_fps = [os.path.join(folder, fn) for fn in images]
    image_fps = [fp.replace('\\', '/') for fp in image_fps]
    image_fps = [fp.replace(' ', '\ ') for fp in image_fps]

    for fp in image_fps:
        out_txt += 'file {}\nduration {}\n'.format(fp, image_duration)


#%%
# out_txt = 'file ' + "\nfile ".join(image_list)

out_dir = os.path.join(song_basedir, 'story')
if not os.path.exists(out_dir): os.mkdir(out_dir)

with open('videos.txt', 'w') as f:
    f.write(out_txt)

import shutil

df_trans_sequence.to_csv(os.path.join(out_dir, 'trans_sequence.csv'))
shutil.move('videos.txt', os.path.join(out_dir, 'videos_story.txt'))

fn_out = 'output_storynx.mov'

os.chdir(out_dir)
os.system('ffmpeg -f concat -safe 0 -i videos_story.txt -c mjpeg -q:v 3 -r {} {}'.format(fps, fn_out))
# %%

