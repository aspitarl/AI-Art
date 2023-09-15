#%%

import os
import pandas as pd
import argparse

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.
gdrive_basedir = os.getenv('base_dir')


USE_DEFAULT_ARGS = False
if USE_DEFAULT_ARGS:
    song = 'spacetrain'
    # scene = 'tram_alien'
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("song")
    args = parser.parse_args()

    song = args.song

# scene_dirs = [fold for fold in os.listdir(allscenes_folder)]
# scene_dirs = [fold for fold in scene_dirs if os.path.isdir(os.path.join(allscenes_folder, fold))]
# scene_dirs = [fold for fold in scene_dirs if 'transitions.csv' in os.listdir(os.path.join(allscenes_folder, fold)) ]


#TODO: change so each song has an input_data file, potnetially two for prompts and transitions. Alternatively just use transitions file in scene folder
output_fp = os.path.join(gdrive_basedir, song, 'all_transitions.csv')

allscenes_folder = os.path.join(gdrive_basedir, song, 'scenes')

dfs = []
for fold in os.listdir(allscenes_folder):

    scene_dir = os.path.join(allscenes_folder, fold)
    if not os.path.isdir(scene_dir):
        continue

    fp_df_transitions = os.path.join(scene_dir, 'transitions.csv')

    if not os.path.exists(fp_df_transitions):
        continue

    print("Copying Transitions from {}".format(fold))

    df = pd.read_csv(fp_df_transitions, index_col=0)

    dfs.append(df)

df_transitions = pd.concat(dfs)

#%%

df_transitions.to_csv(output_fp)

# # https://stackoverflow.com/questions/62618680/overwrite-an-excel-sheet-with-pandas-dataframe-without-affecting-other-sheets
# def write_excel(filename,sheetname,dataframe):
#     with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer: 
#         workBook = writer.book
#         try:
#             workBook.remove(workBook[sheetname])
#         except:
#             print("Worksheet does not exist")
#         finally:
#             dataframe.to_excel(writer, sheet_name=sheetname,index=False)
#             writer.save()

# with pd.ExcelWriter(output_fp, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:



#     df_transitions.to_excel(writer, 'transitions_{}'.format(song))

# # write_excel(output_fp, 'transitions_{song}', df_transitions)
         