cd ..

python -m venv venv

source /home/ubuntu/AI-Art/venv/bin/activate

cd lambda

pip install -r requirements.txt

mkdir -p model_cache
mkdir -p transition_images

git config user.email aspitarte@gmail.com
git config user.name "Lee Aspitarte"