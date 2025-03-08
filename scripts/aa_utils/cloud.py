import os
from PIL import Image
import dotenv; dotenv.load_dotenv()

# Generate functions for pipeline settings, currently for everything other than latent/embeds

def gen_pipe_kwargs_static(row, pipe_name, song_name):
    """
    row - row of df_prompt (prompt_image_definitions)
    """

    pipe_kwargs = {}

    pipe_kwargs['guidance_scale'] = float(row['guidance_scale'])

    if pipe_name == 'controlnet':
        pipe_kwargs['controlnet_conditioning_scale'] = float(row['cnet_scale'])

        mask_name = row['mask']
        pipe_kwargs['image'] = Image.open(os.path.join(os.getenv('meta_dir'), song_name, 'masks', mask_name + '.png'))

    return pipe_kwargs

def gen_pipe_kwargs_transition(t, df_prompt, from_name, to_name, pipe_name, song_name):

    pipe_kwargs = {}

    # directly interpolate with t
    t = float(t)

    # old method
    # The tensor steps are len(num_interpolation_steps) + 1
    # guidance_steps = np.linspace(guidance_scales[0], guidance_scales[1], num_interpolation_steps + 1)
    # cnet_steps = np.linspace(cnet_scales[0], cnet_scales[1], num_interpolation_steps + 1)

    guidance_scales = [
        df_prompt['guidance_scale'][from_name],
        df_prompt['guidance_scale'][to_name]
    ]

    guidance_scale=guidance_scales[0]*(1-t) + guidance_scales[1]*t
    pipe_kwargs['guidance_scale'] = guidance_scale

    if pipe_name == 'controlnet':


        cnet_scales = [
            df_prompt['cnet_scale'][from_name],
            df_prompt['cnet_scale'][to_name]
        ]

        masks = [
            df_prompt['mask'][from_name],
            df_prompt['mask'][to_name]
        ]

        masks = [Image.open(os.path.join(os.getenv('meta_dir'), song_name, 'masks', mask_name + '.png')) for mask_name in masks]
        masks = [m.convert('RGBA') for m in masks]

        mask_interp = Image.blend(masks[0], masks[1], t)
        pipe_kwargs['image'] = mask_interp

        cnet_scale=cnet_scales[0]*(1-t) + cnet_scales[1]*t
        pipe_kwargs['controlnet_conditioning_scale'] = cnet_scale


    return pipe_kwargs



from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

MODEL_CACHE_DIR = os.getenv('model_cache_dir')

def gen_pipe(pipe_name, settings):

    if pipe_name == 'basic':
        pipe = StableDiffusionPipeline.from_pretrained(
                                                        settings['model_string'],
                                                        torch_dtype=torch.float16,
                                                        safety_checker=None,
                                                        cache_dir=MODEL_CACHE_DIR
                                                    )

        pipe = pipe.to("cuda")

        # if one wants to disable `tqdm`
        # https://github.com/huggingface/diffusers/issues/1786
        pipe.set_progress_bar_config(disable=True)

    elif pipe_name == 'controlnet':

        controlnet = ControlNetModel.from_pretrained(
                                                    settings['controlnet_string'], 
                                                    torch_dtype=torch.float32
                                                    )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            settings['model_string'], 
            controlnet=controlnet, 
            torch_dtype=torch.float32, 
            safety_checker=None,
            cache_dir=MODEL_CACHE_DIR
        )

        # if one wants to disable `tqdm`
        # https://github.com/huggingface/diffusers/issues/1786
        pipe.set_progress_bar_config(disable=True)

        from diffusers import UniPCMultistepScheduler

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # this command loads the individual model components on GPU on-demand.
        pipe.enable_model_cpu_offload()

    else:
        raise ValueError(f"Invalid pipe_name: {pipe_name}")

    return pipe
