fps=60

# cd "G:\My Drive\AI Music Visuals Share\New Codes\output_transitions"

cd "G:\My Drive\AI-Art\\$1\scenes\\$2"

mkdir -p rev

input_files="*.mp4"

# echo $input_files

for f in $input_files

do
  echo "Processing $f file..."

  ffmpeg -y -i "$f" -vf reverse "rev/$f"
done
