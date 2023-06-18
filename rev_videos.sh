: ${@?no positional parameters}
source .env

cd "$base_dir\\$1\\transitions"

mkdir -p "$base_dir\\$1\\transitions_rev"

input_files="*.mp4"

# echo $input_files

for f in $input_files

do
  echo "Processing $f file..."

  ffmpeg -n -i "$f" -vf reverse "$base_dir\\$1\\transitions_rev\\$f"
done
