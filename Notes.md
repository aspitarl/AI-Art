
# Notes 

## Movie Format
Looking at revamping movie file structure to allow for quick indexing of movies to automate playing position in TD Movie In TOP with low CPU/GPU usage. I noticed high CPU usage when moving around the index compared to sequential playing, but later realized it didn't happen all the time.

Made this post 

https://forum.derivative.ca/t/movie-file-in-top-not-interpolating-frames-when-specifying-index/374809

Post about codecs 

https://interactiveimmersive.io/blog/beginner/codecs-for-touchdesigner-explained/

> Photo/Motion JPEG: JPEG encoded video, lossy codec and low decode times with medium file sizes. Good for playback both forwards and backwards or for random access.

command used to make a transition: 

`ffmpeg -framerate 5 -i frame%06d.png -c mjpeg -pix_fmt yuv420p -r 5 output_test.mov`

for video text file

`ffmpeg -f concat -safe 0 -i videos_story.txt -c mjpeg -r 5 output_test.mov`

useful article: https://shotstack.io/learn/use-ffmpeg-to-convert-images-to-video/

## Lambda process


### Setup
start lambda instance 

copy ip address to .env and ssh config (TODO: automate or have filetransfer scripts pull from ssh config?)

ssh into instance

clone AI-Art repo 
checkout lambda branch
`cd lambda`
`source setup.sh`

### generate transitions

on local repo

`cd lambda`
`source filetransfer_meta.sh <song_name>`

# 2023-10-29 clip interrogator

https://pypi.org/project/clip-interrogator/0.5.4/

had to change some packages to get clip-interrogator to work. testing with explore prompts cnet to make sure old functionality works

```
  Attempting uninstall: torch
    Found existing installation: torch 2.0.1
    Uninstalling torch-2.0.1:
      Successfully uninstalled torch-2.0.1
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.15.2+cu118
    Uninstalling torchvision-0.15.2+cu118:
      Successfully uninstalled torchvision-0.15.2+cu118
Successfully installed torch-2.0.0 torchvision-0.15.1

```

2.0.1 would probably work but didn't try

had to downgrade transformers

https://github.com/pharmapsychotic/clip-interrogator/issues/62

```
Installing collected packages: transformers
  Attempting uninstall: transformers
    Found existing installation: transformers 4.30.2
    Uninstalling transformers-4.30.2:
      Successfully uninstalled transformers-4.30.2
Successfully installed transformers-4.26.1

```

# 2024-01-24

Returning after break to try and finish emit pipey and cycle. 


## emit
Trying to regenerate scene sequence kv1. Appears transitions downloaded on 12-13 were not completely downloaded (partially black) images causing "png @ 0x chunk too big error " in ffmpeg. Think they would have to be generated as I deleted images from gdrive. Removed those images to another folder 'TI_bad' . Cant remember if they were just extra or completed the graph. Emit movie generation appears to work ok now, but final transitions file is pretty different from kv1. 

## pipey

scene sequence kv_fix is latest and assuming is right. movie generates without issue and new images appear to be full. 


# 2024-04-05 

Going through entire process of making a movie from scratch 

On cloud: Open terminal in `cloud` folder of repo

1. create `song_meta\song_name\tgen_settings.json` and `song_meta\song_name\prompt_image_definitions.csv`.
2. use `cloud/explore_prompts.py` to come up with prompts and seeds and create prompt image definitions. 
  * `python interrogate_CLIP.py song_name -p prompt_name -s seed_name` can be used to interrogate prompts
3. create the prompt image files with `python gen_saved_prompts.py song_name`
4. download prompt_images to local song dir 
  * TODO: add to `cloud\gzip_transitions.sh`

local: google drive folder. song_dir = gdrive_basedir/song_name. Open terminal in `local` folder of repo

4. group images into scenes (`song_dir/scenes/sx`). 
  * local\various\automake_scenes.py` will automate this process based on prompt definition. 
5. Create scene sequence csv and place in `song_dir/prompt_data`. 
7. run `python examine_existing.py song_name` 
6. run `python gen_transitions.py song_name` to random generate transition path
  * TODO: having to run multiple times to get a good sampling of random transitions. Automate, or make a script to generate base level graph, connect one random node to another node in next scene. 
7. upload generated `interscene_transitions.csv`,`intrascene_transitions.csv`, and `existing_transitions.csv` to cloud at `cloud/prompt_data/song_name`


cloud: 
1. run `python gen_transitions.py song_name`
2. run `source gzip_transitions.py` and download tar files and unzip 'here'

local: 

1. run `source gen_ss.sh escape` 