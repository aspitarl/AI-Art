from os.path import join as pjoin
import pandas as pd


import os


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


def load_df_scene_sequence(scene_sequence, song_name):
    scene_sequence_name = "scene_sequence" if scene_sequence == '' else "scene_sequence_{}".format(scene_sequence)
    fp_scene_sequence = pjoin(os.getenv('meta_dir'), song_name, '{}.csv'.format(scene_sequence_name))

    print("loading scene sequence from {}".format(fp_scene_sequence))
    df_scene_sequence = pd.read_csv(fp_scene_sequence , index_col=0)
    return df_scene_sequence