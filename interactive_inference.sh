#!/bin/bash

module load cray-python/3.9.13.1
module load rocm/6.0.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a

# Create a virtual env. and install the following packages
# python3 -m venv .venv
#pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
#pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
source .venv/bin/activate

config_file=./config/AFNO.yaml
config="afno_backbone"
run_num="0"

FCN_DIR="/lustre/orion/cli115/proj-shared/grnydawn/data/fourcastnet"

python3 inference/inference.py \
       --config=${config} \
       --run_num=${run_num} \
       --weights "${FCN_DIR}/model_weights/FCN_weights_v0_org/backbone.ckpt" \
       --override_dir "${FCN_DIR}/output"
