source ../.env

scp -r -i $ssh_key_path $ssh_user@$ssh_ip_address:~/AI-Art/lambda/output/ .