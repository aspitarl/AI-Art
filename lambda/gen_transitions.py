# %%
song_name = 'spacetrain_1024' #@param {type:"string"}
res_height = 564 #@param
res_width = 1024 #@param
create_mp4_files = False #@param {type:"boolean"}


import os
import pandas as pd
import numpy as np
from IPython.display import clear_output

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'
output_basedir = os.path.join('transition_images')
if not os.path.exists(output_basedir): os.mkdir(output_basedir)

# fp = os.path.join(code_folder, 'input_data.xlsx')
# df_prompt = pd.read_excel(fp, 'prompts_{}'.format(song_name), index_col=0).dropna(how='all')

fp = os.path.join('../prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

# df_transitions = pd.read_excel(fp, 'transitions_{}'.format(song_name), dtype={'from_seed': str, 'to_seed': str})
fp = os.path.join('../prompt_data', 'all_transitions.csv')
df_transitions = pd.read_csv(fp, index_col=0).dropna(how='all')

# %%
df_transitions

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

#%%
import torch
from diffusers import StableDiffusionPipeline

# %%
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                               torch_dtype=torch.float16,
                                               safety_checker=None,
                                               cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")

# %%

#TODO: replace below
width = res_width
height = res_height

latent_width = width // 8
latent_height = height // 8


def generate_latent(generator, seed, device='cuda'):

    generator.manual_seed(int(seed))

    latent = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        generator = generator,
        device = device
    )

    return latent

def make_latent_steps(start_latent, stop_latent, steps):
    delta_latent = (stop_latent - start_latent)/steps
    latent_steps = [start_latent + delta_latent*i for i in range(steps + 1)]

    #Check that start and end values are equal to targets within rounding errors
    # assert torch.isclose(latent_steps[0], from_latent, atol=1e-4).all()
    # assert torch.isclose(latent_steps[-1], to_latent, atol=1e-2).all()

    return latent_steps

def get_text_embed(prompt):
    text_input = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

    embed = pipe.text_encoder(text_input.input_ids.to('cuda'))[0]

    return embed

if not os.path.exists(output_basedir): os.makedirs(output_basedir)

# %%
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

# %%
skip_existing = True

generator = torch.Generator(device="cuda")

max_seed_characters = 4 # Take the first few numbers of the seed for the name
num_interpolation_steps = 30
num_inference_steps = 40


T = np.linspace(0.0, 1.0, num_interpolation_steps)

for idx, row in df_transitions.iterrows():
  clear_output(wait=True)

  output_name = "{}-{} to {}-{}".format(
      row['from_name'],
      str(row['from_seed'])[:max_seed_characters],
      row['to_name'],
      str(row['to_seed'])[:max_seed_characters]
      )

  output_dir = os.path.join(output_basedir, output_name)
  output_filepath = os.path.join(output_basedir,  "{}.mp4".format(output_name))


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

  from_latent = generate_latent(generator, seeds[0])
  to_latent = generate_latent(generator, seeds[1])

  from_text_embed = get_text_embed(prompts[0])
  to_text_embed = get_text_embed(prompts[1])

  # The tensor steps are len(num_interpolation_steps) + 1
  # latent_steps = make_latent_steps(from_latent, to_latent, num_interpolation_steps)
  # embed_steps = make_latent_steps(from_text_embed, to_text_embed, num_interpolation_steps)
  guidance_steps = np.linspace(guidance_scales[0], guidance_scales[1], num_interpolation_steps + 1)


  for i, t in enumerate(T):

      print("Transition {} out of {}".format(idx, len(df_transitions)))
      print(output_name)
      print("Frame {}/{}".format(i,num_interpolation_steps))

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

  if create_mp4_files:
      make_video_pyav(output_dir,
                      output_filepath=output_filepath,
                      fps=fps
                      )



# %%



