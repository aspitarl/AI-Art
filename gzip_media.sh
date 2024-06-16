#!/bin/bash

input_dir='media'

mkdir -p 'media_gz'

# Get the current timestamp
# timestamp=$(date +%Y%m%d_%H%M%S)

# Get the current timestamp in PST
timestamp=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)


folder_name=$1

# Create a tarball and then compress it into a .gz file
# filename_out="ti_gz/${song}_transitions_${timestamp}.tar.gz"
# Have to unzip twice with 7zip. TODO: compare size between .tar.gz and .zip
# tar -czf "$filename_out" -C "$dir" transition_images

# Define the output filename
filename_out="media_gz/${folder_name}_${timestamp}.tar"
# Create a tar file without compression
tar -cf "$filename_out" -C "$input_dir/$folder_name" .


# Create a zip file
# filename_out="ti_gz/${song}_transitions_${timestamp}.zip"
# zip -rq "$filename_out" "$dir"transition_images