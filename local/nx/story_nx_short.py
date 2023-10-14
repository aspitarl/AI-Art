

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

USE_DEFAULT_ARGS = False
if USE_DEFAULT_ARGS:
    song = 'cycle_mask'
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    song = args.song

from dotenv import load_dotenv; load_dotenv()
gdrive_basedir = os.getenv('base_dir')
# gdrive_basedir = r"G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle"
input_basedir = os.path.join(gdrive_basedir, '{}\scenes'.format(song))

#%%

dir_transitions = os.path.join(gdrive_basedir, song, 'transition_images')

trans_list = [t for t in os.listdir(dir_transitions) if os.path.isdir(pjoin(dir_transitions,t))]
trans_list = [image_names_from_transition(t) for t in trans_list]

trans_list

#%%

trans_from = [t[0] for t in trans_list]
trans_to = [t[1] for t in trans_list]

nodes = set([*trans_from, *trans_to])

nodes


#%%

scene_dir = pjoin(gdrive_basedir, song, 'scenes')

scene_list = [s for s in os.listdir(scene_dir) if os.path.isdir(pjoin(scene_dir,s))]

if 's_test' in scene_list: scene_list.remove('s_test')

scene_list

#%%
# Make a mapping from file to folder name for each scene folder in scene dir

regex = re.compile("([\S\s]+_\d\d\d\d)\d+.png")

scene_dict = {}
for scene in scene_list:
    scene_dict[scene] = [fn for fn in os.listdir(pjoin(scene_dir, scene)) if fn.endswith('.png')]

    # scene_dict[scene] = [regex.match(fn).groups()[0].replace("_","-") for fn in scene_dict[scene]]

    scene_dict[scene] = [fn.rsplit('_', 1)[0] + '-' + fn.rsplit('_', 1)[1] for fn in scene_dict[scene]]

    # remove the .png extension from each filename with a regular expression

    scene_dict[scene] = [re.sub(r'\.png$', '', fn) for fn in scene_dict[scene]]

    # truncate the digits after each hypen to 4 digits

    scene_dict[scene] = [re.sub(r'-(\d+)$', lambda m: '-' + m.group(1)[:4], fn) for fn in scene_dict[scene]]





# scene_dict.pop('s_test')

scene_dict

#%%

# invert scene_dict to make a mapping from file to folder name

file_to_scene_dict = {}
for scene in scene_list:

    for fn in scene_dict[scene]:
        file_to_scene_dict[fn] = scene


file_to_scene_dict


#%%

# existing_both = set([*nodes, *file_to_scene_dict.keys()])

# existing_both

# nodes = [node for node in nodes if node in existing_both]
# file_to_scene_dict = {k:v for k,v in file_to_scene_dict.items() if k in existing_both}


#%%

G = nx.Graph()

G.add_nodes_from(nodes)
G.add_edges_from(trans_list)



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

file_to_scene_dict

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
plt.savefig(pjoin(gdrive_basedir, song, 'story', 'story_graph_2.png'))
# %%

# build a list with a random element of scene_dict for each key in scene_sequence

scene_sequence = sorted(scene_list)
# scene_sequence = list(scene)

scene_sequence = pd.read_csv(os.path.join(gdrive_basedir, song, 'prompt_data', 'scene_sequence.csv'), index_col=0)['scene'].values.tolist()



# sort this list based on the zero padded number in the scene name, ignoring the first character and periods in the name 

# scene_sequence = sorted(scene_sequence, key=lambda x: int(re.search(r'\d+', x).group()))

# scene_sequence = scene_sequence[0:3]

scene_sequence
#%%
#%%

first_scene = scene_sequence[0]
last_scene = scene_sequence[-1]

# find the most connected node in the first scene

def most_connected_node(G, scene):
    scene_nodes = [n for n in G.nodes() if G.nodes[n]['scene'] == scene]
    scene_G = G.subgraph(scene_nodes)

    # get the node with the highest degree
    node = max(scene_G.degree, key=lambda x: x[1])[0]

    return node

first_node = most_connected_node(G, first_scene)
last_node = most_connected_node(G, last_scene)

path = nx.shortest_path(G, first_node, last_node)

path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]


path_edges

#%%

last_node = path[-1]

# go to a random node in the last scene that is not last_node

last_scene_nodes = [n for n in G.nodes() if G.nodes[n]['scene'] == last_scene]

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
    G.nodes[node]['label'] = i


color_map = []

for node in G:
    if node in path:

        color_map.append('green')
    else:
        color_map.append('red')

nx.draw(G, node_color=color_map, with_labels=True, node_size=50, labels=nx.get_node_attributes(G,'label'))

plt.savefig(pjoin(gdrive_basedir, song, 'story', 'story_graph.png'))

# %%

df_transitions = pd.DataFrame(path_edges, columns=['c1','c2'])


df_transitions

#%%

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
song_basedir = os.path.join(gdrive_basedir, song)

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



# %%
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

