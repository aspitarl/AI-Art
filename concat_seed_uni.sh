

cp videos.txt "G:\My Drive\AI-Art\pipey\transitions\s4\videos.txt"

cd "G:\My Drive\AI-Art\pipey\transitions\s4"

rm output8.mp4

ffmpeg -f concat -safe 0 -i videos.txt -c copy output8.mp4