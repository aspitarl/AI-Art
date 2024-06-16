#%%


import os
import pandas as pd

folder_in = r'G:\.shortcut-targets-by-id\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\emit\transition_images'

# Initialize an empty list to store the results
results = []

# Iterate through the subfolders
for root, dirs, files in os.walk(folder_in):
    # Count the number of image files (assuming they have extensions .jpg, .jpeg, .png)
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)
    
    # Append the folder name and the number of images to the results
    folder_name = os.path.basename(root)
    results.append((folder_name, num_images))

# Convert the results to a DataFrame
df = pd.DataFrame(results, columns=['folder_name', 'num_images'])

# Save the DataFrame to a CSV file
# df.to_csv('image_counts.csv', index=False)
# %%

df['num_images'].value_counts()

#%%

count_values = set(df['num_images'].values)

remove_values = [x for x in count_values if x not in [50]]

remove_values
# %%

# show all folders with number of images in remove_values

df[df['num_images'].isin(remove_values)]
# %%

# take all these folders and move them into a 'removed' folder

import shutil

folder_out = os.path.join(folder_in, 'wrong_num_images')

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

for folder_name in df[df['num_images'].isin(remove_values)]['folder_name']:
    if folder_name in ['wrong_num_images', 'transition_images']:
        continue
    folder_in_full = os.path.join(folder_in, folder_name)
    folder_out_full = os.path.join(folder_out, folder_name)
    shutil.move(folder_in_full, folder_out_full)

# %%
