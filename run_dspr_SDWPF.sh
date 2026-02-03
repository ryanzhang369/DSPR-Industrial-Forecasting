#!/bin/bash

# =========================================================
# DSPR Training Script for SDWPF (Wind Power Forecasting)
# Dataset: KDD Cup 2022 (Turbine 1)
# Target: Active Power (Patv)
# Physics: Aerodynamics & Control (Pitch Angle)
# =========================================================


export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH


DATA_FILE="sdwpf_turbine_1_for_dspr.csv"

PHYSICS_FILE="sdwpf_physics.csv"


TARGET_COL="Patv"    
CONTROL_COL="Pab1"   
RUN_NAME="SDWPF_Turbine1_Run01"
SAVE_DIR="./records"

echo "-----------------------------------------------------"
echo "Step 1: Starting DSPR Training..."
echo "-----------------------------------------------------"

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
  --pred_len 24 \
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


LATEST_RES_DIR=$(ls -td ${SAVE_DIR}/${RUN_NAME}_*/results | head -1)

if [ -z "$LATEST_RES_DIR" ]; then
    echo "Error: Could not find results directory."
    exit 1
fi

echo "-----------------------------------------------------"
echo "Step 2: Starting Physics Evaluation..."
echo "Target Dir: $LATEST_RES_DIR"
echo "-----------------------------------------------------"


python evaluate_physics.py \
  --path "$LATEST_RES_DIR" \
  --DATA_INTERVAL_S 600 \
  --CONTROL_CYCLE_STEPS 3 \
  --PHYSICAL_WINDOW 70 \
  --DA_DEADBAND 0.0001 \
  --EPSILON 1e-7

echo "-----------------------------------------------------"
echo "All Tasks Done. Check $LATEST_RES_DIR/physics_results.json"
echo "-----------------------------------------------------"
