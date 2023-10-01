


#%%



G = nx.Graph()

# add nodes for each image in each scene

for scene in scene_dict:
    G.add_nodes_from(scene_dict[scene], scene=scene)

scene_names = list(scene_dict.keys())

image_from = np.random.choice(scene_dict[scene_names[0]])
for i in range(len(scene_names) - 1):
    scene_from = scene_names[i]
    scene_to = scene_names[i+1]

    # add a random edge between scene_from and scene_to

    # get a random image from scene_from

    image_from = image_to

    # get a random image from scene_to

    image_to = np.random.choice(scene_dict[scene_to])

    G.add_edge(image_from, image_to)



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
ec = nx.draw_networkx_edges(G, pos, alpha=1)
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)

plt.colorbar(nc)
plt.axis('off')



#%%

# existing_both = set([*nodes, *file_to_scene_dict.keys()])

# existing_both

# nodes = [node for node in nodes if node in existing_both]
# file_to_scene_dict = {k:v for k,v in file_to_scene_dict.items() if k in existing_both}

