setup VM with setup.sh
generate `song_meta/prompt_image_definitions.csv`

#TODO: split out these file transfer steps, implement better file transfer
transfer prompt metadata to cloud 
generate images with `gen_saved_prompts.py`
transfer back images

#TODO: script to auto setup scenes and scene sequence file. 
collect images into `scenes/scene_name`
create  `scene_sequence.csv` with order of those scenes specifies
run `gen_all_transitions.sh`

transfer back transition metadata 
geneate transitions `gen_transitions.py`
transfer back transition images

run `story.py`