#%%
import os
import pandas as pd
from aa_utils.sd import image_grid
import dotenv; dotenv.load_dotenv()

song_name = 'cycle_mask' #@param {type:"string"}
res_height = 576 #@param
res_width = 1024 #@param

from PIL import Image
mask_image = Image.open(os.path.join('masks', "cyclist_side.png"))

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

model_cache_dir = os.path.join(os.getenv('REPO_DIR'), 'cloud','model_cache')


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float32, 
    safety_checker=None, 
    cache_dir=model_cache_dir
)

# if one wants to disable `tqdm`
# https://github.com/huggingface/diffusers/issues/1786
pipe.set_progress_bar_config(disable=True)

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()


pipe = pipe.to("cuda")


#%%
fp = os.path.join(os.getenv('REPO_DIR'), 'cloud', 'prompt_data', 'prompt_image_definitions.csv')
df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

name_sel = 'burst_city_blue'
prompt = df_prompt['prompt'][name_sel]
guidance_scale = df_prompt['guidance_scale'][name_sel]

num_inference_steps = 20

# rows X cols of images. Reduce for speed and memory issues.
rows = 2
cols = 2

num_images = rows*cols

# Make new random seeds in a hacky way. TODO: probably a function to generate seeds without making a generator instance.

generator = torch.Generator(device="cuda")
seeds = [generator.seed() for i in range(num_images)]
generator = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

print("Prompt: {}".format(prompt))
print("Seeds: {}".format(seeds))

images = pipe(
    prompt,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=rows*cols,
    width=res_width,
    height=res_height,
    num_inference_steps=num_inference_steps,
    control_guidance_start=0.1,
    control_guidance_end=0.6,          
    controlnet_conditioning_scale=0.8,
    image=mask_image
).images

grid = image_grid(images, rows=rows, cols=cols)

grid
# %%


output_dir = os.path.join('output', 'explore', name_sel)
if not os.path.exists(output_dir): os.makedirs(output_dir)

# remove all files in output_dir

for f in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, f))

grid.save(os.path.join(output_dir, 'image_grid.png'))

for i, image in enumerate(images):
    seed=seeds[i]
    image.save(os.path.join(output_dir, 'image_{}.png'.format(seed)))
# %%
