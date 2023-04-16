# AI Art

Codes related to stable diffusion art


# Seed Multiverse Workflow

in below: 
song = pipey, emit, etc.
sx = s1, s2, etc. 

## Get images
Generate images with PromptExplorer.ipynb
Put in input_data.xlsx prompts_song tab (on Gdrive)
use Generate_saved_Prompts.ipynb

## Generate Transitions
Pick 5 images and put in `GDrive/AI-Art/<song>/scene_input_images/sx
run `gen_scene_transition_file.py song sx`
copy data from transitions.xlsx made in that folder into Gdrive input_data.xlsx transitions_song tab

run Generate_Transitions.ipynb

## Making Seed Multiverse movie

Once transitions are complete, run `reorg_files.py song` to move files into scenes directory. 
run `./rev_videos.sh song sx` to generate reverse videos
run `python seed_multiverse.py song sx`
run `./concat_seed_uni.sh song sx`