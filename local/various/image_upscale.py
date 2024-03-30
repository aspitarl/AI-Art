


folder_in = r'G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\pipey\prompt_images'
folder_out = r'G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\pipey\upscaled_images_4'


import os
from super_image import EdsrModel, ImageLoader, DrlnModel
from PIL import Image

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
# model = DrlnModel.from_pretrained('eugenesiow/drln-bam', scale=2)

files = os.listdir(folder_in)

# pick a random 10 of these images
import random

random.seed(42)
random.shuffle(files)



for file in files[:10]:
    if file.endswith('.png'):
        print("Upscaling", file)
        img = Image.open(os.path.join(folder_in, file))
        inputs = ImageLoader.load_image(img)
        preds = model(inputs)

        # ImageLoader.save_image(preds, os.path.join(folder_out, file))

        ImageLoader.save_compare(inputs, preds, os.path.join(folder_out, file))