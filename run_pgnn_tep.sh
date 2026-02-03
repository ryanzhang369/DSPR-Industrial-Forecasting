#!/bin/bash

# =========================================================
# PG-NN Training Script for TEP (Tennessee Eastman Process)
# Model: TimeMixer Backbone + Physics Conservation Loss
# =========================================================

export CUDA_VISIBLE_DEVICES=0

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
SAVE_DIR="${BASE_DIR}/records/"


TARGET_COL="Reactor_Pressure"      

echo "-----------------------------------------------------"
echo "Starting PG-NN Training on TEP Process..."
echo "Model: PG-NN (TimeMixer + Physics Loss)"
echo "Target: $TARGET_COL"
echo "Data: $DATA_FILE"
echo "-----------------------------------------------------"

python -u main_pgnn.py \
  --data_path "$DATA_FILE" \
  --target "$TARGET_COL" \
  --save_dir "$SAVE_DIR" \
  \
  --features MS \
  --freq t \
  --seq_len 96 \
  --pred_len 24 \
  \
  --batch_size 32 \
  --learning_rate 0.001 \
  --train_epochs 20 \
  --patience 5 \
  \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 2 \
  --n_heads 4 \
  --down_sampling_layers 1 \
  --down_sampling_method avg \
  --dropout 0.05 \
  \
  --lambda_phys 0.1 \
  --gpu 0
