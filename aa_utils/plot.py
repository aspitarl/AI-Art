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

    # Make a color map with a different color for each scene based on the scene of each node

    # create number for each group to allow use of colormap

    # get unique groups

    groups = scene_sequence
    mapping = dict(zip(groups,count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[n]['scene']] for n in nodes]

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
    pos = nx.multipartite_layout(G, subset_key='scene', align='horizontal', scale=1, center=(0,0))
    ec = nx.draw_networkx_edges(G, pos, edge_color= edge_colors, alpha=alphas)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=10, cmap=plt.cm.jet)

    plt.colorbar(nc)
    plt.axis('off')

    # make one label of the scene name positioned at the top of the plot

    for scene in scene_dict:
        scene_center = np.mean([pos[n] for n in scene_dict[scene]], axis=0)

        plt.text(-0.2, scene_center[1], scene, fontsize=10, horizontalalignment='center', verticalalignment='center')

