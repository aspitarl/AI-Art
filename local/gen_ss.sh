# Script to be run after transitions have been generated for a song.

#!/bin/bash

# Initialize the variables
song=$1
scene_sequence=""
output_name=""

# Parse the arguments
while (( "$#" )); do
  case "$1" in
    --ss)
      scene_sequence="$2"
      shift 2
      ;;
    -o)
      output_name="$2"
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

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

# if output name is provided
if [ -n "$output_name" ]
then
echo "Generating movie with output name $output_name"
python gen_movie.py $song -o $output_name
else
echo "generating movie"
python gen_movie.py $song
fi
