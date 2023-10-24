

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


scene_dict = pd.read_csv(os.path.join(gdrive_basedir, args.song, 'prompt_data', 'scene_dict.csv'), index_col=0).to_dict()['0']

# convert the values from strings to lists
scene_dict = {k: v.split(',') for k,v in scene_dict.items()}

# remove single quotes and list brackets from each element in the list
scene_dict = {k: [re.sub(r"['\[\]]", '', fn).strip() for fn in v] for k,v in scene_dict.items()}

# truncate the digits after each hyphen to 4 digits
scene_dict = {scene: [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:4], fn) for fn in scene_dict[scene]] for scene in scene_dict}

scene_dict

#%%

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

scene_sequence = scene_sequence[0:5]

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

N_repeats = 0

# pick a start_node that is a random node in the first scene

start_node = np.random.choice([node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene_sequence[0]])

path = [start_node]

for i, scene in enumerate(scene_sequence):

    # make a subgraph of the nodes in the scene

    scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene]

    scene_graph = G_sel.subgraph(scene_nodes)


    # make a random path of length N_repeats edges starting at start_node

    for j in range(N_repeats):
        path.append(np.random.choice(list(scene_graph.neighbors(path[-1]))))



    if i < len(scene_sequence) - 1:
        # find a path to a node in the next scene

        next_scene_nodes = [node for node in G_sel.nodes if G_sel.nodes[node]['scene'] == scene_sequence[i+1]]

        both_scene_graph = G_sel.subgraph([*scene_nodes, *next_scene_nodes])

        # find the shortest path from the last node in the path to a node in the next scene

        path_to_next_scene = nx.shortest_path(both_scene_graph, path[-1], np.random.choice(next_scene_nodes))

        # add this path to the path

        path.extend(path_to_next_scene[1:])

        # set the start_node for the next scene to be the last node in the path

        start_node = path[-1]


#%%

path

# %%
# draw the graph

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

# color the edges in path red 

path_edges = list(zip(path,path[1:]))

nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)

#%%
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
fps = 10
image_duration = 1/fps

for idx, row in df_trans_sequence.iterrows():

    folder = row['input_image_folder']

    images = [fn for fn in os.listdir(folder) if fn.endswith('.png')]
    images = sorted(images)

    if row['reversed']: images = images[::-1]

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

fn_out = 'output_storynx_long.mov'

os.chdir(out_dir)
os.system('ffmpeg -f concat -safe 0 -i videos_story.txt -c mjpeg -q:v 3 -r {} {}'.format(fps, fn_out))
# %%

