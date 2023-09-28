from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def generate_latent(generator, seed, pipe, latent_height, latent_width, device='cuda'):

    generator.manual_seed(int(seed))

    latent = torch.randn(
        (1, pipe.unet.in_channels, latent_height, latent_width),
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

def get_text_embed(prompt, pipe):
    text_input = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

    embed = pipe.text_encoder(text_input.input_ids.to('cuda'))[0]

    return embed

import numpy as np
import torch

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

