# Create and activate venv
# python -m venv .venv

# install python extension

# choose requirements in reqs
# pip install -r requirements.txt

source .env

# git config user.email $git_email
# git config user.name $git_name

# mkdir -p $media_dir
# mkdir -p $model_cache_dir

pushd scripts

# pip install -e . 

# Downloads models and final test
python image_gen/explore_prompts.py nspiral_test -p sunflower

popd