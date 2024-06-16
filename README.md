# AI Art

Produce movies that smoothly transition from one stable diffusion prompt to another.


# Setup 

These codes output images and movides to a specified directory. Create a file in the base repository directory named `.env` and inside this file add the following line 

```
media_dir=...
meta_dir=...
model_cache_dir...(SD model path, only for cloud)
```

* meta_dir - directory of text files defining output, starting point.
    * prompt_image_definitions.csv
    * tgen_settings.json
    * scene_sequence.csv
* media_dir - working directory of generated files
    * prompt images
    * scene image collections
    * transition_meta (transition_meta)
    * transition_images
    * masks (?)
    * stories
    * story temp folder (?)


The codes were designed on a combination of cloud environment (google cloud) for generation of images, as well as local machine for generating metadata and combining images into movies. The codes for each enviroment as well as python requirments are in respective folders



# Movie generation workflow

in below: 
`<song>`: song name, a subdirectory that images and metadata will be stored in. located in `media_dir/<song>`
`<sx>`: 'scene' which is a collection of related images

The metadata describing prompts and transitions is located in `transition_meta` in the song directory. This folder needs to be transfered to google cloud with 

For image generation scripts ( `explore_prompts.py`, `gen_saved_prompts.py`, and `gen_transitions.py`) there are associated notebooks that can be run on google collab. 

## Create images (cloud)

Put prompt defiintions in `media_dir/<song>/transition_meta/prompt_image_definitions.csv` in desired song folder song. Transfer to the cloud server. 

Use `explore_prompts.py` to find prompts and add seed info to `prompt_image_definitions.csv`

once all files are generated run `gen_saved_prompts.py` to generate all prompts and images 
#TODO: this script and gen transitions need to have <song> added as a command line argument, set at the top of the script for now. 

transfer back images to local

## Generate Transitions

### local

gen_ss.sh has the sequence of steps to make a song

### Folder structure

transition_sequence.csv - final sequence used by ffmpeg to generate a movie
transition_sequence_gen.csv - sequence generated for image creation, i.e. known path through graph