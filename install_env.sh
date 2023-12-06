#!/bin/bash

eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml && conda activate policy_diffusion

# for CUDA 11 and cuDNN 8.2 or newer
wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl
pip install jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl

# for CUDA 11 and cuDNN 8.0.5 or newer, uncomment this and comment out the previous two lines
#wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl
#pip install jaxlib-0.3.25+cuda11.cudnn805-cp39-cp39-manylinux2014_x86_64.whl

pip install jax==0.3.25
pip install jaxopt==0.5.5

rm jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl