: ${@?no positional parameters}
# https://stackoverflow.com/questions/5228345/how-can-i-reference-a-file-for-variables-using-bash
source ../.env

song_name=$1

cd "$base_dir\\$song_name\\scenes"


for f in *; do
    if [ -d "$f" ]; then

    echo $f

    #TODO: How to not have to change diractories. Append all valid scenes to an array? 
    cd "$repo_dir\\scripts"
    python gen_scene_transition_file.py $song_name $f
    cd "$base_dir\\$song_name\\scenes"

    fi
done


cd "$repo_dir\\scripts"
python gen_interscene_transition.py $song_name
python combine_transitions_csvs.py $song_name