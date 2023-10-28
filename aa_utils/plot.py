import matplotlib.pyplot as plt
import networkx as nx

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
