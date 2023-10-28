import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from itertools import count

def plot_path_labels(G_sel, path_edges):
    plt.figure(figsize=(6,6))

    # Add a label to each node in the path
    path = [edge[0] for edge in path_edges]
    labels = {node: i for i, node in enumerate(path)}
    nx.set_node_attributes(G_sel, labels, 'label')

    # Use the same layout as in examine_existing.py
    pos = nx.multipartite_layout(G_sel, subset_key='scene', align='horizontal', scale=1, center=(0,0))

    color_map = ['green' if node in path else 'red' for node in G_sel]

    nx.draw(G_sel, pos=pos, node_color=color_map, with_labels=True, node_size=50, labels=nx.get_node_attributes(G_sel,'label'))


def plot_scene_sequence(G, scene_sequence, scene_dict):
    plt.figure(figsize=(6,10))


    G_plot = G.copy()

    # Fix the scene names so that they sort correctly, e.g. 01_scene1, 02_scene2, etc. 
    # multipartite layout sorts them automatically and a key cannot be passed to the sort function
    for i, scene in enumerate(scene_sequence):
        for node in G_plot.nodes():
            if G_plot.nodes[node]['scene'] == scene:
                # zero pad the number so that it sorts correctly

                i_scene = str(i+1).zfill(2)

                G_plot.nodes[node]['scene'] = '{}_{}'.format(i_scene, scene)

    groups = [G_plot.nodes[n]['scene'] for n in G_plot.nodes()]
    mapping = dict(zip(groups,count()))
    nodes = G_plot.nodes()
    colors = [mapping[G_plot.nodes[n]['scene']] for n in nodes]


    edge_colors = []
    alphas = []

    for edge in G.edges():
        if G.edges[edge]['exists']:
            edge_colors.append('green')
            alphas.append(1)
            # add a new attribute to the edge to indicate that it exists
            G.edges[edge]['Weight'] = 1
        else:
            edge_colors.append('red')
            alphas.append(0.1)
            G.edges[edge]['Weight'] = 0.1

    # drawing nodes and edges separately so we can capture collection for colobar

    # pos = nx.spring_layout(G)
    pos = nx.multipartite_layout(G_plot, subset_key='scene', align='horizontal', scale=1, center=(0,0))
    ec = nx.draw_networkx_edges(G_plot, pos, edge_color= edge_colors, alpha=alphas)
    nc = nx.draw_networkx_nodes(G_plot, pos, nodelist=nodes, node_color=colors, node_size=10, cmap=plt.cm.jet)

    plt.colorbar(nc)
    plt.axis('off')

    for scene in groups:
        nodes = [n for n in G_plot.nodes() if G_plot.nodes[n]['scene'] == scene]    
        for n in nodes:
            if n in pos:
                plt.text(-0.2, pos[n][1], scene, fontsize=10, horizontalalignment='center', verticalalignment='center')
                break



