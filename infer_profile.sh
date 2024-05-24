#!/usr/bin/bash

module load cray-python/3.11.5

. .venv/bin/activate

python3 inference/inference_profile.py \
       --config=afno_backbone \
       --run_num=0 \
       --weights '/pscratch/sd/y/youngsun/data/FourCastNet/model_weights/FCN_weights_v0/backbone.ckpt' \
       --override_dir '/pscratch/sd/y/youngsun/data/FourCastNet/output' 
