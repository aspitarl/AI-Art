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

