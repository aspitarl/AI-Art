# Procedure

Tested on cloud

## VM setup

setup VM with setup.sh (untested)

Make .env file 

```
media_dir='/home/aspitarte/AI-Art/media'
meta_dir='/home/aspitarte/AI-Art/song_meta'
model_cache_dir='/home/aspitarte/AI-Art/model_cache'
```

media is output to `<media_dir>/<song_name>`
song metadata is in `<meta_dir>/<song_name>`

## song workflow

generate `prompt_image_definitions.csv` and setup `tgen_settings.json`
explore and dial in with `image_gen/explore_prompts.py <song_name> -p <prompt_name>`

generate all images with `image_gen/gen_saved_prompts.py <song_name>`

group images into scene folders
    this can be done automatically with `story_gen/automake_scenes.py <song_name>`
    can also manually group images in `scenes/scene_name` folders and create  `scene_sequence.csv` with order of those scenes specifies


run `story_gen/gen_transition_meta.py` <song_name>
run `story_gen/examine_existing_transitions.py` <song_name>

geneate transitions `image_gen/gen_transitions.py <song_name>`

generate path and movie with `story_gen/gen_ss.sh <song_name>`