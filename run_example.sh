#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

run="SDWPF_Turbine1_Run01"
data="sdwpf_turbine_1_processed.csv"
phys="sdwpf_physics_prior.csv"

echo "Preprocessing data..."
python data_provider/industrial_data_preprocessor.py

echo "Running DSPR example..."
python -u main_dspr.py \
  --root_path ./ \
  --data_path "$data" \
  --target Patv \
  --adj_path "$phys" \
  --run_name "$run" \
  --save_dir ./records/ \
  --features MS \
  --freq t \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
  --control_col Pab1 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 5 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --dropout 0.1 \
  --phys_alpha 0.6 \
  --lambda_phys 0.05 \
  --gpu 0

