#!/bin/bash

# =========================================================
# 1. Full DSPR (TimesNet Backbone + GNN + Physics)
# =========================================================

export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
# 注意：这里指向的是 TimesNet 版本的 Python 脚本
PYTHON_SCRIPT="${BASE_DIR}/main_dspr_tn.py" 
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
PHYSICS_FILE="${BASE_DIR}/tep_physics_prior.csv"
SAVE_DIR="${BASE_DIR}/records/"

TARGET_COL="Reactor_Pressure"
CONTROL_COL="Reactor_Cooling_Valve" 
RUN_NAME="TEP_Full_TimesNet_GNN_Phys"

echo "-----------------------------------------------------"
echo "Starting Full Model: DSPR (TimesNet + GNN + Physics)"
echo "-----------------------------------------------------"

python -u "$PYTHON_SCRIPT" \
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
  --batch_size 128 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3 \
  \
  --d_model 32 \
  --d_ff 128 \
  --e_layers 2 \
  --gnn_heads 4 \
  --dropout 0.05 \
  \
  --embed timeF \
  --top_k 3 \
  --num_kernels 6 \
  \
  --phys_alpha 0.5 \
  --lambda_phys 0.1 \
  \
  --gpu 0

# 注意：这里没有 --ablation 参数