import os
import pandas as pd
from PIL import Image
import dotenv; dotenv.load_dotenv()

def load_df_transitions(dir_transition_meta):

    fp = os.path.join(dir_transition_meta, 'intrascene_transitions.csv')
    df_trans_intrascene = pd.read_csv(fp, index_col=0).dropna(how='all')
    fp = os.path.join(dir_transition_meta, 'interscene_transitions.csv')
    df_trans_interscene = pd.read_csv(fp, index_col=0).dropna(how='all')

    df_transitions = pd.concat([df_trans_interscene, df_trans_intrascene])

    df_existing = pd.read_csv(os.path.join(dir_transition_meta, 'existing_transitions.csv'), index_col=0)


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


    # remove transitions that already exist, and print those that were removed 

    len_before = len(df_transitions)

    df_transitions = df_transitions[~df_transitions.index.isin(df_existing.index)]

    len_after = len(df_transitions)

    print("Removed {} transitions that already exist".format(len_before - len_after))

    df_transitions = df_transitions.where(df_transitions['compute'] == 'y').dropna(how='all')
    # Get the number of rows before dropping duplicates
    num_rows_before = df_transitions.shape[0]
    df_transitions = df_transitions.astype({
        'from_name': str,
        'from_seed': int,
        'to_name':str,
        'to_seed':int,
        'compute':str,
        'duration':float

    })

    # Drop duplicates
    df_transitions = df_transitions.drop_duplicates()

    # Get the number of rows after dropping duplicates
    num_rows_after = df_transitions.shape[0]

    # Calculate and print the number of rows dropped
    num_rows_dropped = num_rows_before - num_rows_after
    print(f"Dropped {num_rows_dropped} duplicate rows for transitions")
        
    return df_transitions

def load_df_prompt(song_meta_dir, seed_delimiter=','):
    fp = os.path.join(song_meta_dir, 'prompt_image_definitions.csv')
    df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

    if df_prompt.index.duplicated().any():
        print("Warning: Duplicated prompts found, dropping duplicates")
        print(df_prompt[df_prompt.index.duplicated()].index)
        df_prompt = df_prompt[~df_prompt.index.duplicated()]

    df_prompt = df_prompt.rename(columns={
        'seeds': 'seed_list_str'
    })

    df_prompt = df_prompt.astype({
        'prompt': str,
        'seed_list_str': str,
        'guidance_scale': float
    })

    # Make a new column that is a list of seeds
    df_prompt['seeds'] = df_prompt['seed_list_str'].str.split(seed_delimiter)
    for idx, row in df_prompt.iterrows():
        seeds = row['seeds']
        seeds = [s.strip() for s in seeds]
        seeds = [int(s) for s in seeds]
        df_prompt.at[idx, 'seeds'] = seeds

    return df_prompt

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
