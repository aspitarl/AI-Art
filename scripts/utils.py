import os




def clip_names_from_transition(transition_fp):

    name = os.path.splitext(transition_fp)[0]

    name = os.path.split(name)[1]

    # c1, c2 = re.findall('\d\d\d\d', name)
    c1, c2 = name.split(" to ")

    return c1, c2


def transition_fn_from_transition_row(row, max_seed_characters=4):
    output_name = "{}-{} to {}-{}.mp4".format(
    row['from_name'],
    str(row['from_seed'])[:max_seed_characters],
    row['to_name'],
    str(row['to_seed'])[:max_seed_characters]
    )

    return output_name

def clip_names_from_transition_row(row, max_seed_characters=4):
    c1 = "{}-{}".format(
    row['from_name'],
    str(row['from_seed'])[:max_seed_characters])

    c2 = "{}-{}".format(
    row['to_name'],
    str(row['to_seed'])[:max_seed_characters]
    )

    return c1, c2

import re
def extract_seed_prompt_fn(fn, regex = re.compile("(\S+)_(\d+).png")):
    """
    returns the prompt and seed string from an image filename
    """

    m = re.match(regex, fn)

    if m:
        prompt = m.groups()[0]
        seed = m.groups()[1]
        return prompt, seed
    else:
        return None, None
    

import pandas as pd

def gendf_imagefn_info(fns_images):
    df = pd.DataFrame(
    fns_images,
    index = range(len(fns_images)),
    columns = ['fn']
)

    df[['prompt', 'seed']] = df.apply(lambda x: extract_seed_prompt_fn(x['fn']), axis=1, result_type='expand')
    return df

#Transition sequence dataframe utility functions #TODO: extract components as functions

import numpy as np

def gendf_trans_sequence(df_transitions, num_output_rows):
    # Generate the sequence of transitions in terms of clip index
    seed_lookup = gen_seed_lookup(df_transitions)
    num_videos = len(seed_lookup)

    def find_next_idx(cur_idx):
        valid_idxs = [i for i in range(num_videos) if i != cur_idx]
        next_idx = np.random.randint(0,num_videos-1)
        next_idx = valid_idxs[next_idx]
        return next_idx

    cur_idx = 0

    df_trans_sequence = pd.DataFrame(columns = ['c1','c2'], index = list(range(num_output_rows)))

    trans_names_forward = (df_transitions['c1'] + df_transitions['c2']).values
    trans_names_rev = (df_transitions['c2'] + df_transitions['c1']).values

    for i in df_trans_sequence.index:  
        found_match = False

        for j in range(1000):
            next_idx = find_next_idx(cur_idx)

            cur_name = seed_lookup[cur_idx]
            next_name = seed_lookup[next_idx]

            checkstr = cur_name + next_name
            if checkstr in trans_names_forward or checkstr in trans_names_rev:
                found_match = True
                break


        if not found_match:
            raise ValueError("could not find match")
        
        df_trans_sequence['c1'][i] = seed_lookup[cur_idx]
        df_trans_sequence['c2'][i] = seed_lookup[next_idx]

        cur_idx=next_idx


    df_trans_sequence
    return df_trans_sequence


def gen_seed_lookup(df_transitions):
    # Make a lookup table for each clip
    all_cs = list(set([*df_transitions['c1'], *df_transitions['c2']]))
    c_id = list(range(len(all_cs)))

    seed_lookup = pd.Series(all_cs, index=c_id, name='seed_str')

    return seed_lookup
