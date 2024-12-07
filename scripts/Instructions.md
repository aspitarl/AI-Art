# Procedures

## Full automated workflow

`run_all.sh` has sequence of steps to go from a `song_name` (folder name) in `song_meta` folder. There are two test songs, nspiral_test (regular SD), and escape_test (Controlnet). Test that everything is working with 

`source run_all.sh nspiral_test`
`source run_all.sh escape_test`

## Movie Generation Workflow Steps

### Explore Images

generate `prompt_image_definitions.csv` and setup `tgen_settings.json`
explore and dial in with `python image_gen/explore_prompts.py <song_name> -p <prompt_name>`

### Automated Pipeline

once `prompt_image_definitions` and `tgen_settings.json` are created, the entire pipeline can be run wiht `source run_all.sh <song_name>`

### Pipeline Steps

generate all images with `python image_gen/gen_saved_prompts.py <song_name>`

group images into scene folders
    this can be done automatically with `python story_gen/automake_scenes.py <song_name>`
    can also manually group images in `scenes/scene_name` folders and create  `scene_sequence.csv` with order of those scenes specifies


run `python story_gen/gen_transition_meta.py` <song_name>
run `python story_gen/examine_existing_transitions.py` <song_name>

geneate transitions `python image_gen/gen_transitions.py <song_name>`

generate path and movie with `story_gen/gen_ss.sh <song_name>`



## File structure

media is output to `<media_dir>/<song_name>`
song metadata is in `<meta_dir>/<song_name>`

* meta_dir - directory of text files defining output, starting point.
    * prompt_image_definitions.csv
    * tgen_settings.json
    * scene_sequence.csv
    * masks 
* media_dir - working directory of generated files
    * prompt images
    * scene image collections
    * transition_meta 
    * transition_images
    * stories
    * story temp folder 

transition_sequence.csv - final sequence used by ffmpeg to generate a movie
transition_sequence_gen.csv - sequence generated for image creation, i.e. known path through graph
