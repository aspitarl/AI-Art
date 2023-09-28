: ${@?no positional parameters}
source ../.env

cd "$base_dir\\$song_name"
scp -r -i $ssh_key_path $ssh_user@$ssh_ip_address:~/AI-Art/cloud/output/ .