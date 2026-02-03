#!/bin/bash

# =========================================================
# DSPR Training Script for SDWPF (Wind Power Forecasting)
# Dataset: KDD Cup 2022 (Turbine 1)
# Target: Active Power (Patv)
# Physics: Aerodynamics & Control (Pitch Angle)
# =========================================================

# 1. 设置环境
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH

# 2. 定义文件路径
# 确保 convert_sdwpf_v2.py 生成的数据在当前目录下
DATA_FILE="sdwpf_turbine_1_for_dspr.csv"
# 确保之前生成的物理图文件在当前目录下
PHYSICS_FILE="sdwpf_physics.csv"

# 3. 核心配置
TARGET_COL="Patv"    # 预测目标: 有功功率
CONTROL_COL="Pab1"   # 控制变量: 桨叶角度 (Pitch Angle)
RUN_NAME="SDWPF_Turbine1_Run01"

echo "-----------------------------------------------------"
echo "Starting DSPR Training on SDWPF Wind Turbine..."
echo "Target: $TARGET_COL"
echo "Physics Prior: $PHYSICS_FILE"
echo "Data File: $DATA_FILE"
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
  --seq_len 96 \
  --label_len 0 \
  --pred_len 12 \
  \
  --control_col $CONTROL_COL \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 15 \
  --patience 5 \
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

# 参数变更说明:
# --freq 10t: 适配数据的10分钟采样间隔。
# --batch_size 64: 数据量有3.5万行，比TEP大，增大Batch更稳定。
# --down_sampling_layers 2: 风电波动包含高频和低频，2层下采样能更好提取趋势。
# --train_epochs 15: 数据量大，稍微多训练几轮。
