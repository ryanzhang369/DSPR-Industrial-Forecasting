#!/bin/bash

# =========================================================
# DSPR Training Script for Rotary Kiln Dataset
# Dataset: Rotary Kiln (10s Interval)
# Target: C1CO
# Physics: Process Coupling & Gas Composition
# =========================================================

# 1. 设置环境
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

# 2. 定义文件路径
# 你的回转窑数据文件（CSV）
DATA_FILE="6-hzy-2601-04-10s-filter.csv"

# 物理先验邻接矩阵（如无可先留空或用经验构造）
PHYSICS_FILE="rotary_kiln_physics.csv"

# 3. 核心配置
TARGET_COL="C1CO"          # 预测目标：C1CO
CONTROL_COL="gwfjzs"     # 控制变量（如风量/阀门/给料，按实际可替换）
RUN_NAME="RotaryKiln_C1CO_Run01"

echo "-----------------------------------------------------"
echo "Starting DSPR Training on Rotary Kiln Dataset..."
echo "Target: $TARGET_COL"
echo "Control Variable: $CONTROL_COL"
echo "Physics Prior: $PHYSICS_FILE"
echo "Data File: $DATA_FILE"
echo "Time Interval: 10s"
echo "-----------------------------------------------------"

# 4. 运行命令
python -u main_dspr.py \
  --root_path ./ \
  --data_path $DATA_FILE \
  --target $TARGET_COL \
  --adj_path $PHYSICS_FILE \
  --run_name $RUN_NAME \
  --save_dir ./records/ \
  \
  --features MS \
  --freq t \
  --seq_len 16 \
  --label_len 0 \
  --pred_len 4 \
  \
  --control_col $CONTROL_COL \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --train_epochs 15 \
  --patience 3 \
  \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --dropout 0.1 \
  \
  --phys_alpha 0.6 \
  --lambda_phys 0.05 \
  --gpu 0
