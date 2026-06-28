#!/bin/bash
export PATH=/data/anaconda3/envs/econverify/bin:$PATH
cd /home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main
mkdir -p logs attention_data
export CUDA_VISIBLE_DEVICES=5
for dp in 30 40 50; do
  echo "=== Extracting dp=$dp ==="
  python -u extract_attn.py --data_pool $dp --output_dir ./attention_data 2>&1 | tee logs/extract_dp${dp}.log
  echo "=== dp=$dp done ==="
done
echo "ALL DONE"
