#!/bin/bash
# Check if an argument was provided



song_name=$1

story_dir="G:/.shortcut-targets-by-id/1Dpm6bJCMAI1nDoB2f80urmBCJqeVQN8W/AI-Art Kyle/$song_name/stories"

pushd "$story_dir"

# If videos.txt exists, remove it
if [ -f "$story_dir/videos_concat.txt" ]; then
echo "Removing old videos_concat.txt"
  echo "" >  "$story_dir/videos_concat.txt"
fi

concatenated_videos=""
# Iterate over each subdirectory of story_dir
for dir in "$story_dir"/*; do
  # Reset concatenated_videos to an empty string
#   concatenated_videos=""

  # If the subdirectory contains a videos.txt file, concatenate its contents
  if [ -f "$dir/videos.txt" ]; then
    concatenated_videos+=$(cat "$dir/videos.txt")
    concatenated_videos+=$'\n'
  fi

done
# Output the concatenated contents to videos_concat.txt in the subdirectory
echo "$concatenated_videos" > "videos_concat.txt"


fps=10

# # Run the ffmpeg command
ffmpeg -f concat -safe 0 -i videos_concat.txt -y -c mjpeg -q:v 3 -r $fps "$story_dir/movies_combined.mov"

popd