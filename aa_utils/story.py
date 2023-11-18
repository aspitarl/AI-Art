
# Add missing imports 

import os
import re
import pandas as pd
import networkx as nx



def downselect_to_scene_sequence(G, scene_sequence):
    # Remove all edges from the graph that do not connect nodes in the scene_sequence
    # Make a list of all edges in the graph
    all_edges = list(G.edges)

    # Only keep edges that connect nodes with pairs of scenes that are adjacent in the scene_sequence
    adjacent_edges = []
    for i in range(len(scene_sequence)-1):
        scene1 = scene_sequence[i]
        scene2 = scene_sequence[i+1]

        # Keep edges that connect nodes in these two scenes in either direction
        adjacent_edges.extend([(u,v) for u,v in all_edges if G.nodes[u]['scene'] == scene1 and G.nodes[v]['scene'] == scene2])
        adjacent_edges.extend([(u,v) for u,v in all_edges if G.nodes[u]['scene'] == scene2 and G.nodes[v]['scene'] == scene1])

    # Also include edges that connect nodes in the same scene
    adjacent_edges.extend([edge for edge in all_edges if G.nodes[edge[0]]['scene'] == G.nodes[edge[1]]['scene']])

    # Downselect the graph to only include these edges and their nodes
    G_sequence = G.edge_subgraph(adjacent_edges)
    
    return G_sequence


def gen_path_edges_short(G_sel, scene_sequence):
    first_scene = scene_sequence[0]
    last_scene = scene_sequence[-1]

    def most_connected_node(G_sel, scene):
        scene_nodes = [n for n in G_sel.nodes() if G_sel.nodes[n]['scene'] == scene]
        scene_G_sel = G_sel.subgraph(scene_nodes)

        # get the node with the highest degree
        node = max(scene_G_sel.degree, key=lambda x: x[1])[0]

        return node

    #TODO: probably need to check that there is a path between these nodes
    first_node = most_connected_node(G_sel, first_scene)
    last_node = most_connected_node(G_sel, last_scene)

    path = nx.shortest_path(G_sel, first_node, last_node)

    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]

    return path_edges





def generate_text_for_ffmpeg(df_transitions, fps):
    out_txt = ''
    image_duration = 1/fps
    for _, row in df_transitions.iterrows():
        folder = row['input_image_folder']
        images = sorted([fn for fn in os.listdir(folder) if fn.endswith('.png')])
        if row['reversed']:
            images = images[::-1]
        images = images[:-1]
        image_fps = [os.path.join(folder, fn) for fn in images]
        image_fps = [fp.replace('\\', '/') for fp in image_fps]
        image_fps = [fp.replace(' ', '\ ') for fp in image_fps]
        for fp in image_fps:
            out_txt += f"file {fp}\nduration {image_duration}\n"

    return out_txt



def generate_output_video(fps, out_dir, output_filename):
    os.chdir(out_dir)
    os.system(f"ffmpeg -f concat -safe 0 -i videos.txt -y -c mjpeg -q:v 3 -r {fps} {output_filename}")
