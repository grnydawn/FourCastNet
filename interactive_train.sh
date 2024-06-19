#!/bin/bash

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# Create a virtual env. and install the following packages
# python3 -m venv .venv
# pip install torch torchvision h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
source .venv/bin/activate

ngpu=16 # 4
config_file=./config/AFNO.yaml
config="afno_backbone"
#run_num="check"
#run_num="monitor"
run_num="profile"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node 4 bash -c "source export_DDP_vars.sh && $cmd"
