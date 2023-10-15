import os



def image_names_from_transition(transition_name):

    # c1, c2 = name.split(" to ")

    #TODO: This could be used to separate out 
    # regex = "(.+?)-(\d+) to (.+?)-(\d+)"
    regex = "(.+?-\d+) to (.+?-\d+)"

    m = re.match(regex, transition_name)
    im1, im2 = m.groups()

    return im1, im2


def transition_fn_from_transition_row(row, max_seed_characters=4):
    output_name = "{}-{} to {}-{}.mp4".format(
    row['from_name'],
    str(row['from_seed'])[:max_seed_characters],
    row['to_name'],
    str(row['to_seed'])[:max_seed_characters]
    )

    return output_name

def clip_names_from_transition_row(row, max_seed_characters=4):
    c1 = "{}-{}".format(
    row['from_name'],
    str(row['from_seed'])[:max_seed_characters])

    c2 = "{}-{}".format(
    row['to_name'],
    str(row['to_seed'])[:max_seed_characters]
    )

    return c1, c2

import re
def extract_seed_prompt_fn(fn, regex = re.compile("([\S\s]+)_(\d+).png")):
    """
    returns the prompt and seed string from an image filename
    """

    m = re.match(regex, fn)

    if m:
        prompt = m.groups()[0]
        seed = int(m.groups()[1])
        return prompt, seed
    else:
        return None, None
    

import pandas as pd

def gendf_imagefn_info(fns_images):
    df = pd.DataFrame(
    fns_images,
    index = range(len(fns_images)),
    columns = ['fn']
)

    df[['prompt', 'seed']] = df.apply(lambda x: extract_seed_prompt_fn(x['fn']), axis=1, result_type='expand')
    return df

#Transition sequence dataframe utility functions #TODO: extract components as functions

import numpy as np

def find_next_idx(cur_idx, num_videos):
    valid_idxs = [i for i in range(num_videos) if i != cur_idx]
    next_idx = np.random.randint(0,num_videos-1)
    next_idx = valid_idxs[next_idx]
    return next_idx


def gendf_trans_sequence(df_transitions, num_output_rows, start_clip=None, end_clip=None, max_iter=1000):
    # Generate the sequence of transitions in terms of clip index
    seed_lookup = gen_seed_lookup(df_transitions)
    num_videos = len(seed_lookup)


    if start_clip:
        cur_idx = seed_lookup[seed_lookup == start_clip].index[0]
    else:
        cur_idx = 0

    df_trans_sequence = pd.DataFrame(columns = ['c1','c2'], index = list(range(num_output_rows)))

    trans_names_forward = (df_transitions['c1'] + df_transitions['c2']).values
    trans_names_rev = (df_transitions['c2'] + df_transitions['c1']).values
    valid_transitions = [*trans_names_forward, *trans_names_rev]

    for i in df_trans_sequence.index:  

        if (i == len(df_trans_sequence) -1) and end_clip:
            df_trans_sequence['c1'][i] = seed_lookup[cur_idx]
            df_trans_sequence['c2'][i] = end_clip
            break

        found_match = False

        for j in range(max_iter):
            next_idx = find_next_idx(cur_idx, num_videos)

            cur_name = seed_lookup[cur_idx]
            next_name = seed_lookup[next_idx]

            checkstr = cur_name + next_name
            if checkstr in valid_transitions:
                found_match = True
                break


        if not found_match:
            raise ValueError("could not find match from clip: {}".format(cur_name))
        
        df_trans_sequence['c1'][i] = seed_lookup[cur_idx]
        df_trans_sequence['c2'][i] = seed_lookup[next_idx]

        cur_idx=next_idx

    return df_trans_sequence


def gen_seed_lookup(df_transitions):
    # Make a lookup table for each clip
    all_cs = list(set([*df_transitions['c1'], *df_transitions['c2']]))
    c_id = list(range(len(all_cs)))

    seed_lookup = pd.Series(all_cs, index=c_id, name='seed_str')

    return seed_lookup


import networkx as nx

def find_path_edges(G, scene_names, N_repeats, node_from=None):

    path_edges = []

    if node_from is None: node_from = np.random.choice([n for n in G.nodes() if G.nodes[n]['scene'] == scene_names[0]])
    node_to = None

    for i in range(len(scene_names)):
        scene_from = scene_names[i]

        for j in range(N_repeats):
            # get a random node from scene_from

            if node_to is not None:
                # print('test')
                node_from = node_to

            # TODO: this is a hack to get the most connected node in the scene, need to add kwarg to find_path_edges to specify this

            # # get a random node from scene_to
            # node_to = np.random.choice([n for n in G.nodes() if G.nodes[n]['scene'] == scene_from])

            # go to the most connected node in scene_from

            scene_nodes = [n for n in G.nodes() if G.nodes[n]['scene'] == scene_from]
            scene_G = G.subgraph(scene_nodes)

            # get the node with the highest degree
            # node_to = max(scene_G.degree, key=lambda x: x[1])[0]

            # get a random node
            
            node_to = np.random.choice([n for n in scene_G.nodes()])
            

            # find a path between the two nodes
            path = nx.shortest_path(G, node_from, node_to)

            # add the path to the list of path edges
            path_edges.extend([(path[i], path[i+1]) for i in range(len(path)-1)])

        if i < len(scene_names) - 1:
            
            scene_to = scene_names[i+1]
            # get a random node from scene_from
            
            if node_to is not None:
                node_from = node_to

            # get a random node from scene_to
            node_to = np.random.choice([n for n in G.nodes() if G.nodes[n]['scene'] == scene_to])

            # find a path between the two nodes
            path = nx.shortest_path(G, node_from, node_to)

            # add the path to the list of path edges
            path_edges.extend([(path[i], path[i+1]) for i in range(len(path)-1)])

    return path_edges