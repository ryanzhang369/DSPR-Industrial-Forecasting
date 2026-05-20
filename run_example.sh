#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset LD_LIBRARY_PATH

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
run="SDWPF_Turbine1_Run01"
raw_sdwpf="wtbdata_245days.csv"
raw_tep="TEP_FaultFree_Training.RData"
data="sdwpf_turbine_1_processed.csv"
phys="sdwpf_physics_prior.csv"

save_dir="./records/"

mkdir -p "$save_dir"

echo "Running DSPR example on SDWPF..."

"$PYTHON" -u main_dspr.py \
  --run_name "$run" \
  --root_path ./ \
  --data_path "$data" \
  --target Patv \
  --features MS \
  --adj_path "$phys" \
  --control_col Pab1 \
  --save_dir "$save_dir" \
  --freq t \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 48 \
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

echo "Done. Results are saved under: $save_dir"
