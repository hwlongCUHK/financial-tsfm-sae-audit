#!/bin/bash
export PATH=/data/anaconda3/envs/econverify/bin:$PATH
cd /home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main
mkdir -p logs diffusion_checkpoints

CUDA_VISIBLE_DEVICES=2 python -u attention_diffusion.py --data_pool 30 --epochs 200 > logs/e1_dp30.log 2>&1 &
PID30=$!
CUDA_VISIBLE_DEVICES=3 python -u attention_diffusion.py --data_pool 40 --epochs 200 > logs/e1_dp40.log 2>&1 &
PID40=$!
CUDA_VISIBLE_DEVICES=4 python -u attention_diffusion.py --data_pool 50 --epochs 200 > logs/e1_dp50.log 2>&1 &
PID50=$!

echo "dp30=$PID30 dp40=$PID40 dp50=$PID50"
wait
echo "E1 ALL DONE"
