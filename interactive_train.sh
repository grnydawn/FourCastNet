#!/bin/bash

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn
#export NCCL_DEBUG=info
#export NCCL_PROTO=Simple
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

module load cray-python/3.9.13.1
module load rocm/6.0.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a

# Create a virtual env. and install the following packages
# python3 -m venv .venv
#pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
#pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
source .venv/bin/activate

ngpu=32 # 4
config_file=./config/AFNO.yaml
config="afno_backbone"
#run_num="check"
#run_num="profile"
run_num="profile_trace"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu bash -c "source export_DDP_vars.sh && $cmd"
