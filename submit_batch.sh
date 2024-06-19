#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --account=atm112
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -J afno
#SBATCH -o afno_backbone_finetune.out

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn
#export NCCL_DEBUG=info
#export NCCL_PROTO=Simple
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=$(hostname)

module load cray-python/3.9.13.1
module load rocm/6.0.0
module load amd-mixed/6.0.0
module load craype-accel-amd-gfx90a

source .venv/bin/activate

# copy pre-trained weights for finetune
cp -f \
	/lustre/orion/cli115/proj-shared/grnydawn/data/fourcastnet/FCN_weights_v0/backbone.ckpt \
	/lustre/orion/cli115/proj-shared/grnydawn/data/fourcastnet/FCN_ERA5_data_v0/output/training_checkpoints/best_ckpt.tar

# Create a virtual env. and install the following packages
# python3 -m venv .venv
#pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
#pip install h5py matplotlib ruamel.yaml timm einops scipy torch-tb-profiler
set -x
srun -u \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
