#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH -C gpu
#SBATCH --account=m4259_g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J afno
###SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o afno_backbone_finetune.out

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

module load cray-python/3.11.5

. .venv/bin/activate

set -x
srun -u --mpi=pmi2 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
