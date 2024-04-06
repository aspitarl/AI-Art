import os
from PIL import Image
from clip_interrogator import Config, Interrogator


import argparse

parser = argparse.ArgumentParser(description='Interrogate a CLIP model with an image.')

parser.add_argument('song_name', type=str, help='The name of the song to generate transitions for')
parser.add_argument('--prompt_name', '-p', type=str, default="treestory_la",
                    help='Name of the prompt to interrogate.')

parser.add_argument('--starting_seed', '-s', type=str, default="3962",
                    help='Seed of the image to interrogate.')


args = parser.parse_args()
song_name = args.song_name

# image_folder = "cnet_scripts/output/explore/{}".format(args.prompt_name)
image_folder = os.path.join('output', song_name, 'prompt_images')

# find the first image with starting_seed in the filename

image_path = None

for f in os.listdir(image_folder):
    if args.starting_seed in f:
        image_path = os.path.join(image_folder, f)
        break


"""
CLIP Interrogator uses OpenCLIP which supports many different pretrained CLIP models. For the best prompts for Stable Diffusion 1.X use ViT-L-14/openai for clip_model_name. For Stable Diffusion 2.0 use ViT-H-14/laion2b_s32b_b79k
"""

image = Image.open(image_path).convert('RGB')

# image = image.resize((224, 224))


ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", cache_path="model_cache"))
print(ci.interrogate(image))