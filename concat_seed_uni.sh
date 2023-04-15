

cp videos.txt "G:\My Drive\AI-Art\pipey\scenes\\$1\videos.txt"

cd "G:\My Drive\AI-Art\pipey\scenes\\$1"

rm concatenated_$1.mp4

ffmpeg -f concat -safe 0 -i videos.txt -c copy concatenated_$1.mp4

cd ..

mv "$1/concatenated_$1.mp4" "conatenated_$1.mp4"