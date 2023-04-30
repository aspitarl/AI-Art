: ${@?no positional parameters}
# https://stackoverflow.com/questions/5228345/how-can-i-reference-a-file-for-variables-using-bash
source .env

cd "$base_dir\\$1\scenes\\$2"

# rm concatenated_$1.mp4

ffmpeg -f concat -safe 0 -i videos.txt -c copy concatenated_$2.mp4

cd ..

mv "$2/concatenated_$2.mp4" "conatenated_$2.mp4"