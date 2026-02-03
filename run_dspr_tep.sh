#!/bin/bash

# =========================================================
# DSPR Training Script for TEP (Tennessee Eastman Process)
# Target: Reactor Pressure
# Physics: Ideal Gas Law & Mass Balance
# =========================================================

# 1. 设置环境
export CUDA_VISIBLE_DEVICES=0
# unset LD_LIBRARY_PATH  # 如果你在特定服务器环境下遇到库冲突，可取消注释

# 2. 定义文件路径 (使用绝对路径更安全)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
PHYSICS_FILE="${BASE_DIR}/tep_physics_prior.csv"
SAVE_DIR="${BASE_DIR}/records/"

# 检查文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "[Error] 数据文件未找到: $DATA_FILE"
    exit 1
fi

# 3. 核心配置
TARGET_COL="Reactor_Pressure"       # 预测目标
CONTROL_COL="Reactor_Cooling_Valve" # 控制变量 (DSPR会将其作为外部输入 u 处理)
RUN_NAME="TEP_DSPR_Physics_Run01"

echo "-----------------------------------------------------"
echo "Starting DSPR Training on TEP Process..."
echo "Target: $TARGET_COL"
echo "Physics Prior: $PHYSICS_FILE"
echo "Seed: 2024 (Fixed)"
echo "-----------------------------------------------------"

# 4. 运行命令
# 注意：增加了 TimeMixer 必须的参数 (embed, top_k, etc.)
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