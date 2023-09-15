# AI Art

Codes related to stable diffusion art


# Setup 

Create a file in the base repository directory named `.env` and inside this file add the following line 

`base_dir="path/to/your/AI-Art/Gdrive/folder"`

# Seed Multiverse Workflow

in below: 
song = pipey, emit, etc.
sx = s1, s2, etc. 

## Get images
Generate images with PromptExplorer.ipynb

Put promp defiintions in `prompt_image_definitions.csv` in desired song folder song
use Generate_saved_Prompts.ipynb

## Generate Transitions
Pick images and put in `GDrive/AI-Art/<song>/scenes/<scene_name>
TODO: script to Auto make these folders by prompt name

After images are in scene folder
run `gen_scene_transition_file.py song sx`

After doing this for all scenes run `combine_transitions_csvs.py song` to incorporate all exisiting scene transitions csvs into a new csv file `all_transitions.csv` in the song folder. 

#TODO: script to run two above for all scenes in a song. 


run Generate_Transitions.ipynb

Run `./rev_videos.sh` to generate reverse videos for all transitions. This no longer overwrites files so to replace, manually delete. Note there will be red 'already exists' overwrite error text for existing files.

## Making Seed Multiverse movie

running `make_seed_multiverse.sh song sx num_output_rows` should run all of the below automatically

run `python seed_multiverse.py song sx num_output_rows`
run `./concat_seed_uni.sh song sx`