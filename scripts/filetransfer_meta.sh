: ${@?no positional parameters}
source ../.env

song_name=$1

ssh-agent

# scp -i $ssh_key_path "$base_dir\\$song_name\\interscene_transitions.csv" $ssh_user@$ssh_ip_address:~/AI-Art/lambda/prompt_data/interscene_transitions.csv
# scp -i $ssh_key_path "$base_dir\\$song_name\\intrascene_transitions.csv" $ssh_user@$ssh_ip_address:~/AI-Art/lambda/prompt_data/intrascene_transitions.csv
# scp -i $ssh_key_path "$base_dir\\$song_name\\prompt_image_definitions.csv" $ssh_user@$ssh_ip_address:~/AI-Art/lambda/prompt_data/prompt_image_definitions.csv

# # From cloud
# scp -r -i $ssh_key_path $ssh_user@$ssh_ip_address:~/AI-Art/lambda/prompt_data/ "$base_dir\\$song_name"

# to cloud #TODO: why have to cd? 
cd "$base_dir\\$song_name"
scp -r -i $ssh_key_path prompt_data $ssh_user@$ssh_ip_address:~/AI-Art/lambda/