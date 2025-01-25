# %%
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from aa_utils.local import build_graph_simple, gen_path_sequence_fullG
from aa_utils.plot import plot_scene_sequence

from dotenv import load_dotenv; load_dotenv(override=True)
# %%

s_prompt_def = """
name,prompt,seeds,guidance_scale
sunflower,"center of sunflower, close up, yellow petals, blue sky",1000 1001 1002 1003,7.5
nautilus2,"photograph, conch nautilus shell close up on beach, spiral pattern shell",1000 1001 1002 1003,7.5
galaxy,spiral galaxy,1000 1001 1002 1003,7.5
"""

# read in a csv file with the prompts

from io import StringIO
df_prompt = pd.read_csv(StringIO(s_prompt_def))

s_ss = """
,scene,duration,section,start
2,galaxy,3,1,
1,nautilus2,3,1,
0,sunflower,3,1,
"""

df_scene_sequence = pd.read_csv(StringIO(s_ss))


import networkx as nx



G = build_graph_simple(df_scene_sequence,df_prompt)

nx.draw(G)
#%%

gen_path_sequence_fullG(G, df_scene_sequence)
#%%
