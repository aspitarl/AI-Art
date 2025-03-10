source ../.env

lambda_api_key=$lambda_api_key

response_file=response.json

region_search="California"

curl -X GET "https://cloud.lambdalabs.com/api/v1/instance-types" \
    -H 'accept: application/json' \
    -u $lambda_api_key: -o $response_file

jq_command='.data | to_entries[] | select(.value.regions_with_capacity_available[].description | contains("'$region_search'"))'
jq "$jq_command" $response_file
