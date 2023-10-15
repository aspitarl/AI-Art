# AI Art

Produce movies that smoothly transition from one stable diffusion prompt to another.


# Setup 

These codes output images and movides to a specified directory. Create a file in the base repository directory named `.env` and inside this file add the following line 

```
base_dir="path/to/your/AI-Art/Gdrive/folder"
REPO_DIR="path/to/this/repository"
```

The codes were designed on a combination of cloud environment (google cloud) for generation of images, as well as local machine for generating metadata and combining images into movies. The codes for each enviroment as well as python requirments are in respective folders

# Movie generation workflow

in below: 
`<song>`: song name, a subdirectory that images and metadata will be stored in. located in `base_dir/<song>`
`<sx>`: 'scene' which is a collection of related images

The metadata describing prompts and transitions is located in `prompt_data` in the song directory. This folder needs to be transfered to google cloud with 

For image generation scripts ( `explore_prompts.py`, `gen_saved_prompts.py`, and `gen_transitions.py`) there are associated notebooks that can be run on google collab. 

## Create images (cloud)

Put prompt defiintions in `base_dir/<song>/prompt_data/prompt_image_definitions.csv` in desired song folder song. Transfer to the cloud server. 

Use `explore_prompts.py` to find prompts and add seed info to `prompt_image_definitions.csv`

once all files are generated run `gen_saved_prompts.py` to generate all prompts and images 
#TODO: this script and gen transitions need to have <song> added as a command line argument, set at the top of the script for now. 

transfer back images to local

## Generate Transitions

### local
Pick images and put in `GDrive/AI-Art/<song>/scenes/<scene_name>
TODO: script to Auto make these folders by prompt name

create `prompt_data/scene_sequence.csv` with the order of desired scenes and duration (duration is only used iwht `story.py`)

After images are in scene folder
run `gen_all_transitions.sh <song>` to generate intra and interscene transitions and transfer to cloud

### cloud

run `generate_transitions.py`

transfer back transitions to local computer

## Making Transition story

run `story.py <song>`