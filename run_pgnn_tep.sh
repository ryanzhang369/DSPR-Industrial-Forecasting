#!/bin/bash

# =========================================================
# PG-NN Training Script for TEP (Tennessee Eastman Process)
# Model: TimeMixer Backbone + Physics Conservation Loss
# =========================================================

# 1. 设置环境
export CUDA_VISIBLE_DEVICES=0

# 2. 定义文件路径
# 获取当前脚本所在目录，确保路径正确
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_FILE="${BASE_DIR}/tep_reactor_pressure_target.csv"
SAVE_DIR="${BASE_DIR}/records/"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "[Error] 数据文件不存在: $DATA_FILE"
    echo "请先运行数据处理脚本: python process_tep_data.py"
    exit 1
fi

# 3. 核心配置
TARGET_COL="Reactor_Pressure"      
# 注意: PG-NN 会自动将除 Target 外的列作为特征

echo "-----------------------------------------------------"
echo "Starting PG-NN Training on TEP Process..."
echo "Model: PG-NN (TimeMixer + Physics Loss)"
echo "Target: $TARGET_COL"
echo "Data: $DATA_FILE"
echo "-----------------------------------------------------"

# 4. 运行命令
# 注意：
# 1. 移除了 --run_name 和 --label_len，因为 Python 脚本的 parser 中未定义
# 2. pred_len 设为 24 (3min * 24 = 72min)，涵盖更长的物理瞬态
# 3. lambda_phys 设为 0.1，与代码默认值一致，可根据物理约束的强弱进行调整

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