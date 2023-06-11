import os




def clip_names_from_transition(transition_fp):

    name = os.path.splitext(transition_fp)[0]

    name = os.path.split(name)[1]

    # c1, c2 = re.findall('\d\d\d\d', name)
    c1, c2 = name.split(" to ")

    return c1, c2