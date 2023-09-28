: ${@?no positional parameters}
# https://stackoverflow.com/questions/5228345/how-can-i-reference-a-file-for-variables-using-bash
source ../.env

song_name=$1

cd "$repo_dir\\scripts"
python gen_interscene_transitions.py $song_name
python gen_intrascene_transitions.py $song_name