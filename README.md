# DSPR-Industrial-Forecasting

This repository contains the official PyTorch implementation for the paper: **"DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting"**.

## 📖 Introduction

Forecasting complex industrial systems (e.g., chemical kinetics, energy meteorology) faces a fundamental **accuracy-fidelity dilemma**: deep learning models often achieve high statistical precision but violate conservation laws or causal logic.

To bridge this gap, we propose **DSPR**, a trustworthy forecasting framework that shifts the paradigm of physics integration from passive soft constraints (loss penalties) to **active architectural inductive biases**.

## 🚀 Key Features

* 
**Dual-Stream Decomposition:** Explicitly decouples system dynamics into a stable **Trend Stream** (modeling global evolution) and a **Physics-Residual Stream** (capturing regime-dependent deviations).


* 
**Adaptive Window Mechanism:** A novel module that learns flow-dependent transport delays, automatically adjusting the receptive field based on operating conditions.


* 
**Physics-Guided Dynamic Graph:** Disentangles causal topology from spurious correlations by embedding domain knowledge directly into the network structure.


* 
**Pareto-Optimal Performance:** Achieves state-of-the-art accuracy while maintaining near-ideal physical consistency (Mean Conservation Accuracy > 99%).

# Datasets

Due to licensing and file size constraints, raw datasets are not included in this repository. 
Please download them manually:

1. **TEP**: Download `TEP_FaultFree_Training.RData` from [[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1#)].
2. **SDWPF**: Download `wtbdata_245days.csv` from [[Baidu AI Studio](https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a448e2c04a7499c43e1ae4cc28fbdee8e087e2385?responseContentDisposition=attachment%3B%20filename%3Dwtbdata_245days.csv&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-05T14%3A17%3A03Z%2F-1%2F%2F5932bfb6aa3af1bcfb467bf2a4a6877f8823fe96c6f4fd0d4a3caa722354e3ac)].

Place the files in this directory before running the preprocessing script.


## 🧪 Experiments

The model is evaluated on four diverse physical benchmarks covering distinct industrial characteristics:

* **SCR System** (Chemical Kinetics)
* **Rotary Kiln** (Thermodynamics)
* 
**Tennessee Eastman Process (TEP)** (Process Control) 


* 
**SDWPF** (Wind Power Fluid Dynamics) 



## 📊 Results

DSPR establishes a new performance frontier compared to baselines like TimeMixer, PatchTST, and iTransformer. It significantly reduces forecasting error while ensuring the predictions respect physical laws (mass balance, monotonicity, etc.).

| Model | MAE (SCR) | MCA (Conservation) | TVR (Fidelity) |
| --- | --- | --- | --- |
| **DSPR (Ours)** | **0.265** | **99.8%** | **97.2%** |
| TimeMixer | 0.286 | 99.1% | 88.5% |
| PatchTST | 0.297 | 97.9% | 65.4% |

(Data sourced from Table 2 in the paper )

## 🔗 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{zhang2025dspr,
  title={DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting},
  author={Zhang, Yeran and Department of Data Science, City University of Hong Kong},
  booktitle={Conference Proceedings},
  year={2025}
}

```
