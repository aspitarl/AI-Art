import os
import fnmatch

folder = r'G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\emit\transition_images'

for root, dirnames, filenames in os.walk(folder):
    for filename in fnmatch.filter(filenames, '*(1)*'):
        match = os.path.join(root, filename)
        print(match)
        break  # stop the loop after the first match


# import os
# import fnmatch

# folder = r'G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\emit\transition_images'

# for root, dirnames, filenames in os.walk(folder):
#     for filename in fnmatch.filter(filenames, '*(1)*'):
#         match = os.path.join(root, filename)
#         os.remove(match)  # delete the file