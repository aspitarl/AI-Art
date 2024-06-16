song_name=$1

scripts=(
    "image_gen/gen_saved_prompts.py" 
    "story_gen/automake_scenes.py" 
    "story_gen/gen_transition_meta.py" 
    "story_gen/examine_existing_transitions.py" 
    "image_gen/gen_transitions.py"
    )  

for script_name in "${scripts[@]}"; do

    echo $script_name
    python $script_name $song_name

done

pushd story_gen

source gen_ss.sh $song_name

popd