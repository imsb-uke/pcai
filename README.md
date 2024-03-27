# PCAI


## Description

This repo collects code to run ML analysis on PCa images and develops the PCAI environment.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/repo
cd repo

# create conda environment
conda create -n env-pcai python=3.10 -y
conda activate env-pcai

# install libgcc for openslide
conda install -c anaconda glib
conda install -c conda-forge gcc=12.1.0

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install apt requirements
xargs sudo apt-get install -y < requirements.apt


# install python requirements
pip install -r requirements.txt
```

Create Database:

```bash
# create database
python src/create_db.py
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

# Dockerfile

To run the code inside a docker container, the folder [docker](docker) contains the relevant files.

```bash
# build dockerfile:
docker build -f docker/Dockerfile . -t pcai-experiments

# run docker container:
docker run -it -v ~/spt/pcai-pipeline:/app/ pcai-experiments

# build and start with docker-compose:
cd docker
docker-compose build
docker-compose up
```