movie_dir = r'C:\Users\aspit\Downloads\output_transitions\movies'

import os
files = os.listdir(movie_dir)

import re

regex = re.compile("rainbow spiral wave(\d)-(\d+) to rainbow spiral wave(\d)-(\d+).mp4")

groups = []
matching_files = []
for file in files:
    g = regex.match(file)

    if g is not None:
        gs = g.groups()
        groups.append((gs[0] + '_' + gs[1], gs[2] + '_' + gs[3]))
        matching_files.append(file)


current_index = 0
new_order = [0]

for i in range(len(groups)):
    current_g = groups[current_index]
    target = current_g[1]

    for j, g in enumerate(groups):
        if target == g[0]:
            if j not in new_order:

                new_order.append(j)
                current_index = j
                break

new_file_order = [matching_files[i] for i in new_order]

new_paths = [os.path.join(movie_dir, f) for f in new_file_order]

with open('output/file_order.txt', 'w') as f:
    for path in new_paths:
        f.write("file '{}'\n".format(path))


