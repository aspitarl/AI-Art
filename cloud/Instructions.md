# Procedure

Tested on cloud

## VM setup

setup VM with setup.sh (untested)

Make .env file 

```
media_dir='/home/aspitarte/AI-Art/output'
meta_dir='/home/aspitarte/AI-Art/song_meta'
model_cache_dir='/home/aspitarte/AI-Art/cloud/model_cache'
```

media is output to `<media_dir>/<song_name>`
song metadata is in `<meta_dir>/<song_name>`

## song workflow

generate `prompt_image_definitions.csv` and setup `tgen_settings.json`
explore and dial in with `cloud/explore_prompts.py <song_name> -p <prompt_name>`

generate all images with `cloud/gen_saved_prompts.py <song_name>`

group images into scene folders
    this can be done automatically with `local/various/automake_scenes.py <song_name>`
    can also manually group images in `scenes/scene_name` folders and create  `scene_sequence.csv` with order of those scenes specifies


run `local/gen_transitions.py` <song_name>
run `local/examine_existing.py` <song_name>

geneate transitions `cloud/gen_transitions.py <song_name>`

generate path and movie with `local/gen_ss.sh <song_name>`