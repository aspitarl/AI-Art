#%%
import os

if not os.path.exists('model_cache'): os.mkdir: 'model_cache'

song_name = 'spacetrain_1024' #@param {type:"string"}
res_height = 576 #@param
res_width = 1024 #@param

# code_folder = '/content/gdrive/MyDrive/AI-Art Lee'

# fp = os.path.join(code_folder, 'input_data.xlsx')
# df_prompt = pd.read_excel(fp, 'prompts_{}'.format(song_name), index_col=0).dropna(how='all')
# df_prompt

# fp = os.path.join(code_folder, song_name, 'prompt_image_definitions.csv')
# df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                               torch_dtype=torch.float16,
                                               safety_checker=None,
                                               cache_dir='model_cache'
                                               )


pipe = pipe.to("cuda")




#%%
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


# # https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds

# prompt_name = 'spring forest2'
# prompt = df_prompt['prompt'][prompt_name]
# guidance_scale = float(df_prompt['guidance_scale'][prompt_name])


prompt_components = [
    "a floating tram",
    "descending into an alien world",
    "Matte Painting",
    "retrowave color scheme",
    "purple orange yellow maroon",
    "full view of vehicle",
]

prompt = ", ".join(prompt_components)

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

images = pipe(prompt, generator=generator, num_images_per_prompt=rows*cols, width=res_width, height=res_height).images

grid = image_grid(images, rows=rows, cols=cols)

grid
# %%

images[0].save('test.png')