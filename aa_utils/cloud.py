import os
import pandas as pd
import dotenv; dotenv.load_dotenv()

def load_df_transitions(dir_prompt_data):

    fp = os.path.join(dir_prompt_data, 'intrascene_transitions.csv')
    df_trans_intrascene = pd.read_csv(fp, index_col=0).dropna(how='all')
    fp = os.path.join(dir_prompt_data, 'interscene_transitions.csv')
    df_trans_interscene = pd.read_csv(fp, index_col=0).dropna(how='all')

    df_transitions = pd.concat([df_trans_interscene, df_trans_intrascene])

    df_existing = pd.read_csv(os.path.join(dir_prompt_data, 'existing_transitions.csv'), index_col=0)


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

def load_df_prompt(song_meta_dir):
    fp = os.path.join(song_meta_dir, 'prompt_image_definitions.csv')
    df_prompt = pd.read_csv(fp, index_col=0).dropna(how='all')

    if df_prompt.index.duplicated().any():
        print("Warning: Duplicated prompts found, dropping duplicates")
        print(df_prompt[df_prompt.index.duplicated()].index)
        df_prompt = df_prompt[~df_prompt.index.duplicated()]

    df_prompt = df_prompt.astype({
        'prompt': str,
        'seeds': str,
        'guidance_scale': float
    })

    return df_prompt

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

MODEL_CACHE_DIR = os.path.join(os.getenv('REPO_DIR'), 'cloud', 'model_cache')

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
