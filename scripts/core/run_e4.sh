#!/bin/bash
export PATH=/data/anaconda3/envs/econverify/bin:$PATH
cd /home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main
mkdir -p logs results
for dp in 30 40 50; do
  echo "=== E4 dp=$dp ==="
  CUDA_VISIBLE_DEVICES=5 python -u e4_compare.py --data_pool $dp --T_sample 50 2>&1 | grep -v 'UserWarning\|Precomputed\|Triggered\|Warning:'
done
echo "E4 ALL DONE"
