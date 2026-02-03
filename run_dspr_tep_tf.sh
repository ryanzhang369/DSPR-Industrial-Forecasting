#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${BASE_DIR}/main_dspr_tf.py" 
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
PHYSICS_FILE="${BASE_DIR}/tep_physics_prior.csv"
SAVE_DIR="${BASE_DIR}/records/"

TARGET_COL="Reactor_Pressure"
CONTROL_COL="Reactor_Cooling_Valve" 
RUN_NAME="TEP_Ablation_TimeFilterOnly"  # 名字改一下以便区分

echo "-----------------------------------------------------"
echo "Starting Ablation Study: Tuning to Match TSlib"
echo "-----------------------------------------------------"

python -u "$PYTHON_SCRIPT" \
  --root_path "$BASE_DIR" \
  --data_path "$DATA_FILE" \
  --target "$TARGET_COL" \
  --adj_path "$PHYSICS_FILE" \
  --run_name "TEP_Ablation_MatchTSlib" \
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
  --train_epochs 10 \
  --patience 3 \
  \
  --d_model 32 \
  --d_ff 128 \
  --e_layers 2 \
  --n_heads 8 \
  --gnn_heads 4 \
  --dropout 0.05 \
  \
  --embed timeF \
  --top_k 5 \
  --num_kernels 6 \
  \
  --patch_len 16 \
  --top_p 0.5 \
  --model_alpha 0.1 \
  \
  --ablation \
  --gpu 0