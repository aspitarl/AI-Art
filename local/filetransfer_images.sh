: ${@?no positional parameters}
source ../.env

song_name=$1

# cd "$base_dir"

scp -r -i $ssh_key_path $ssh_user@$ssh_ip_address:~/AI-Art/cloud/output/$song_name/ "$base_dir"