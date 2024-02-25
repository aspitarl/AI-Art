# Script to be run after transitions have been generated for a song.

#!/bin/bash

# Assign the first positional argument to the variable song
song=$1

# Assign the second positional argument to the variable scene_sequence
scene_sequence=$2

# if scene sequence not provide use default
if [ -z "$scene_sequence" ]
then
echo "Examining existing transitions for $song"
python examine_existing.py $song 

echo "generating story"
python story_nx_sections.py $song 

else

echo "Examining existing transitions for $song"
python examine_existing.py $song --ss $scene_sequence

echo "generating story"
python story_nx_sections.py $song --ss $scene_sequence

fi

echo "generating movie"
python gen_movie.py $song 