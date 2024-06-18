#!/bin/bash

# Create a virtual env. and install the following packages
# python3 -m venv .venv
# pip install torch torchvision h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
source .venv/bin/activate

config_file=./config/AFNO.yaml
config="afno_backbone"
run_num="0"

FCN_DIR="/pscratch/sd/y/youngsun/data/FourCastNet"

python3 inference/inference.py \
       --config=${config} \
       --run_num=${run_num} \
       --weights "${FCN_DIR}/model_weights/FCN_weights_v0_org/backbone.ckpt" \
       --override_dir "${FCN_DIR}/output"

