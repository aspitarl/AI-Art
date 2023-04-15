fps=60

# cd "G:\My Drive\AI Music Visuals Share\New Codes\output_transitions"




cd "G:\My Drive\AI-Art\pipey\scenes"

cd $1

ls

input_files="*.mp4"

for f in $input_files

do
if ! (test -f $f); then
  echo "Processing $f file..."

  ffmpeg -y -i "$f" -vf reverse "rev/$f"
fi

done
