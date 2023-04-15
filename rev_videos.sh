fps=60

# cd "G:\My Drive\AI Music Visuals Share\New Codes\output_transitions"



cd "G:\My Drive\AI-Art\pipey\transitions\s4"


# ls

input_files="*.mp4"

for f in $input_files

do

if ! (test -f $f); then
  echo "Processing $f file..."
#   # take action on each file. $f store current file name

#   # ffmpeg \
#   #   -i "$f" \
#   #   -y \
#   #   -crf 10 \
#   #   -vf "minterpolate=fps=$fps:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
#   #   "smoothed_videos/$f.mp4"

echo $f
ffmpeg -y -i "$f" -vf reverse "rev/$f"

# break
fi

done
