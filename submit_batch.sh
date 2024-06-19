#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m4259_g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J afno
#SBATCH -o afno_backbone_finetune.out

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

module load cray-python/3.11.5

source .venv/bin/activate

cp -f \
	/pscratch/sd/y/youngsun/data/FourCastNet/model_weights/FCN_weights_v0_org/backbone.ckpt \
	/pscratch/sd/y/youngsun/data/FourCastNet/FCN_weights_v0/output/training_checkpoints/best_ckpt.tar	

set -x
srun -u --mpi=pmi2 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
