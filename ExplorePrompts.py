# %% [markdown]
# # Prompt Explorer
# 
# This is a notebook for exploring prompts and seeds. Random seeds are generated and displayed. The idea is to have this running alongside other more complicated notebooks, but this one doesn't needs to have gdrive connection. Then when finding cool prompt/seed combos, but them in the prompts google sheet and have those more complex notebooks reference the sheet vs defining prompts as variables in the code. 

# %%
print('hello')

# %%

import torch
from diffusers import StableDiffusionPipeline

# %%

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               torch_dtype=torch.float16,
                                               safety_checker=None
                                               )  


pipe = pipe.to("cuda")
