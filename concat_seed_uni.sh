: ${@?no positional parameters}
# https://stackoverflow.com/questions/5228345/how-can-i-reference-a-file-for-variables-using-bash
source .env



# cd "G:.shortcut-targets-by-id\\1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W\AI-Art Kyle\\$1\scenes\\$2"

cd "$base_dir\\$1\scenes\\$2"


ls
# rm concatenated_$1.mp4

ffmpeg -f concat -safe 0 -i videos.txt -c copy concatenated_$2.mp4

cd ..

mv "$2/concatenated_$2.mp4" "conatenated_$2_4.mp4"