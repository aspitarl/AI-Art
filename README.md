# Stable Exaverse

Produce movies that represent a random but controlled traversal through a networkx graph where the nodes are stable diffusion images and the edges are transitions between images in the latent space. The nodes are grouped in to a sequence of subgraphs, called 'scenes' and the path through the graph will move from scene to scene through a 'scene sequence'. This repository is a series of scripts (test on Google Cloud) that start from a set of csv metadata files to generate a movie corresponding this path. 

See `scripts/Instructions.md` for instructions. 