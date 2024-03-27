# %%
song_name = 'window' #@param {type:"string"}
res_height = 564 #@param
res_width = 1024 #@param


import os
import pandas as pd
import numpy as np
from IPython.display import clear_output
from aa_utils.sd import generate_latent, get_text_embed, slerp
import torch
from diffusers import StableDiffusionPipeline

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'
output_basedir = os.path.join('output', song_name, 'transition_images')
if not os.path.exists(output_basedir): os.makedirs(output_basedir)

fp = os.path.join('prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

fp = os.path.join('prompt_data', 'intrascene_transitions.csv')
df_trans_intrascene = pd.read_csv(fp, index_col=0).dropna(how='all')
fp = os.path.join('prompt_data', 'interscene_transitions.csv')
df_trans_interscene = pd.read_csv(fp, index_col=0).dropna(how='all')

df_transitions = pd.concat([df_trans_interscene, df_trans_intrascene])

df_existing = pd.read_csv(os.path.join('prompt_data', 'existing_transitions.csv'), index_col=0)


#%%

def get_output_name(row, max_seed_characters=4):
    output_name = "{}-{} to {}-{}".format(
      row['from_name'],
      str(row['from_seed'])[:max_seed_characters],
      row['to_name'],
      str(row['to_seed'])[:max_seed_characters]
      )
    
    return output_name

# make a new column that is the string returned from get_output_name

df_transitions['output_name'] = [get_output_name(row) for idx, row in df_transitions.iterrows()]
df_existing['output_name'] = [get_output_name(row) for idx, row in df_existing.iterrows()]

df_transitions = df_transitions.set_index('output_name')
df_existing = df_existing.set_index('output_name')



# %%

# remove transitions that already exist, and print those that were removed 

len_before = len(df_transitions)

df_transitions = df_transitions[~df_transitions.index.isin(df_existing.index)]

len_after = len(df_transitions)

print("Removed {} transitions that already exist".format(len_before - len_after))



# %%
df_transitions = df_transitions.where(df_transitions['compute'] == 'y').dropna(how='all')

df_transitions = df_transitions.astype({
    'from_name': str,
    'from_seed': int,
    'to_name':str,
    'to_seed':int,
    'compute':str,
    'duration':float

})

if df_prompt.index.duplicated().any():
  print("Warning: Duplicated prompts found, dropping duplicates")
  print(df_prompt[df_prompt.index.duplicated()].index)
  df_prompt = df_prompt[~df_prompt.index.duplicated()]

df_prompt = df_prompt.astype({
    'prompt': str,
    'seeds': str,
    'guidance_scale': float
})

df_transitions

# %%
pipe = StableDiffusionPipeline.from_pretrained(
                                                "stabilityai/stable-diffusion-2-1",
                                                # "CompVis/stable-diffusion-v1-4",
                                                torch_dtype=torch.float16,
                                                safety_checker=None,
                                                cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")

# if one wants to disable `tqdm`
# https://github.com/huggingface/diffusers/issues/1786
pipe.set_progress_bar_config(disable=True)


# %%
skip_existing = True

generator = torch.Generator(device="cuda")

max_seed_characters = 4 # Take the first few numbers of the seed for the name
num_interpolation_steps = 30
num_inference_steps = 40


T = np.linspace(0.0, 1.0, num_interpolation_steps)

from tqdm import tqdm

for i_row, (idx, row) in enumerate(df_transitions.iterrows()):
  clear_output(wait=True)

  output_name = row.name

  output_dir = os.path.join(output_basedir, output_name)

  if os.path.exists(output_dir):
      if skip_existing:
          print("{} already exists, skipping".format(output_name))
          continue
      else:
          print("{} already exists, deleting images".format(output_name))
          for fn in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, fn))
  else:
      if not os.path.exists(output_dir): os.makedirs(output_dir)

  prompts = [
      df_prompt['prompt'][row['from_name']],
      df_prompt['prompt'][row['to_name']]
      ]

  guidance_scales = [
      df_prompt['guidance_scale'][row['from_name']],
      df_prompt['guidance_scale'][row['to_name']]
  ]

  seeds = [row['from_seed'], row['to_seed']]

  duration = row['duration']
  fps = num_interpolation_steps/duration


  latent_width = res_width // 8
  latent_height = res_height // 8

  from_latent = generate_latent(generator, seeds[0], pipe, latent_height, latent_width)
  to_latent = generate_latent(generator, seeds[1], pipe, latent_height, latent_width)

  from_text_embed = get_text_embed(prompts[0], pipe)
  to_text_embed = get_text_embed(prompts[1], pipe)

  # The tensor steps are len(num_interpolation_steps) + 1
  # latent_steps = make_latent_steps(from_latent, to_latent, num_interpolation_steps)
  # embed_steps = make_latent_steps(from_text_embed, to_text_embed, num_interpolation_steps)
  guidance_steps = np.linspace(guidance_scales[0], guidance_scales[1], num_interpolation_steps + 1)


  print("Transition {} out of {}".format(i_row, len(df_transitions)))
  print(output_name)
  for i, t in enumerate(tqdm(T)):

      embeds = torch.lerp(from_text_embed, to_text_embed, t)
      # latents = torch.lerp(from_latent, to_latent, t)
      latents = slerp(float(t), from_latent, to_latent)

      with torch.autocast('cuda'):
        images = pipe(
            prompt_embeds=embeds,
            guidance_scale=guidance_steps[i],
            latents = latents,
            num_inference_steps = num_inference_steps
        )

      clear_output(wait=True)

      output_image = images.images[0]

      output_number_string = str(i).zfill(6)
      output_image.save(os.path.join(output_dir, "frame{}.png".format(output_number_string)))


# %%



