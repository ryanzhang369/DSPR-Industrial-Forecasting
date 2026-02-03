#!/bin/bash

# =========================================================
# DSPR Training Script for TEP (Tennessee Eastman Process)
# Target: Reactor Pressure
# Physics: Ideal Gas Law & Mass Balance
# =========================================================


export CUDA_VISIBLE_DEVICES=0

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
PHYSICS_FILE="${BASE_DIR}/tep_physics_prior.csv"
SAVE_DIR="${BASE_DIR}/records/"


TARGET_COL="Reactor_Pressure"      
CONTROL_COL="Reactor_Cooling_Valve" 
RUN_NAME="TEP_DSPR_Physics_Run01"

echo "-----------------------------------------------------"
echo "Starting DSPR Training on TEP Process..."
echo "Target: $TARGET_COL"
echo "Physics Prior: $PHYSICS_FILE"
echo "Seed: 2024 (Fixed)"
echo "-----------------------------------------------------"


python -u main_dspr.py \
  --root_path "$BASE_DIR" \
  --data_path "$DATA_FILE" \
  --target "$TARGET_COL" \
  --adj_path "$PHYSICS_FILE" \
  --run_name "$RUN_NAME" \
  --save_dir "$SAVE_DIR" \
  --seed 2024 \
  \
  --features MS \
  --freq t \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 6 \
  --control_col "$CONTROL_COL" \
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
  --gnn_heads 4 \
  --dropout 0.05 \
  \
  --phys_alpha 0.5 \
  --lambda_phys 0.1 \
  \
  --down_sampling_layers 1 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  \
  --embed timeF \
  --decomp_method moving_avg \
  --moving_avg 25 \
  --channel_independence 1 \
  --top_k 5 \
  --num_kernels 6 \
  --gpu 0
