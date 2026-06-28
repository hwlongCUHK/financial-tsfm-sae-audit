#!/bin/bash
export PATH=/data/anaconda3/envs/econverify/bin:$PATH
cd /home/houwanlong/marketing_kg/asset_allocation/Signature-Informed-Transformer-For-Asset-Allocation-main
mkdir -p logs

echo "Launching dp=30 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  --is_training 1 --root_path ./asset_data/ --data_path full_dataset.csv \
  --model_id dp30 --model Signature --data FULL --data_pool 30 \
  --window_size 60 --horizon 20 --d_model 8 --n_heads 8 --num_layers 1 \
  --hidden_c 64 --ff_dim 64 --sig_input_dim 2 --cross_sig_dim 1 \
  --des Exp --itr 3 > logs/e0_dp30.log 2>&1 &

echo "Launching dp=40 on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  --is_training 1 --root_path ./asset_data/ --data_path full_dataset.csv \
  --model_id dp40 --model Signature --data FULL --data_pool 40 \
  --window_size 60 --horizon 20 --d_model 8 --n_heads 4 --num_layers 2 \
  --hidden_c 8 --ff_dim 32 --sig_input_dim 2 --cross_sig_dim 1 \
  --des Exp --itr 3 > logs/e0_dp40.log 2>&1 &

echo "Launching dp=50 on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python -u run.py \
  --is_training 1 --root_path ./asset_data/ --data_path full_dataset.csv \
  --model_id dp50 --model Signature --data FULL --data_pool 50 \
  --window_size 60 --horizon 20 --d_model 8 --n_heads 8 --num_layers 2 \
  --hidden_c 64 --ff_dim 8 --sig_input_dim 2 --cross_sig_dim 1 \
  --des Exp --itr 3 > logs/e0_dp50.log 2>&1 &

sleep 5
echo "Processes:"
ps aux | grep "run.py" | grep -v grep
