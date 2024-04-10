#!/bin/bash

song_name=$1
run_name=$2

python gen_saved_prompts.py $song_name -o $run_name
python plot_grid.py $song_name -i $run_name