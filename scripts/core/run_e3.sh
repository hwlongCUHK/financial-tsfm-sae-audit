#!/bin/bash
export PATH=/data/anaconda3/envs/econverify/bin:$PATH
cd /home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main
mkdir -p logs diffusion_checkpoints
CUDA_VISIBLE_DEVICES=2 python -u e3_sig_diffusion.py --data_pool 30 > logs/e3_dp30.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -u e3_sig_diffusion.py --data_pool 40 > logs/e3_dp40.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python -u e3_sig_diffusion.py --data_pool 50 > logs/e3_dp50.log 2>&1 &
wait
echo "E3 ALL DONE"
