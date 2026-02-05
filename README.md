# DSPR-Industrial-Forecasting

This repository contains the official PyTorch implementation for the paper:

**"DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting"**.

## 📖 Introduction

Forecasting complex industrial systems (e.g., chemical kinetics, energy meteorology) requires balancing statistical precision with physical plausibility. Standard deep learning models often achieve low error rates (MSE/MAE) but fail to respect fundamental conservation laws or causal logic—a phenomenon we term **"fidelity collapse."**

**DSPR (Dual-Stream Physics-Residual Network)** addresses this by shifting physics integration from passive loss penalties (like in PINNs) to **active architectural inductive biases**. Instead of forcing a single model to learn everything, DSPR explicitly decouples the workflow:

1. **Trend Stream**: Handles high-energy, inertial patterns using a statistical base forecaster.
2. **Residual Stream**: Focuses purely on regime-dependent deviations (transients) using physics-guided graph structures and adaptive delays.

## 🏗️ Methodology

![DSPR Architecture](figures/fig2.png)
*Figure 1: **The dual-stream architecture of DSPR.** The Statistical Stream captures global trends, while the Physics-Aware Stream explicitly models regime-dependent residuals through adaptive delays and dynamic graphs.*

The DSPR framework addresses non-stationarity in industrial systems by structurally decoupling dynamics into two orthogonal components: a stable **Statistical Trend Stream** and a regime-dependent **Physics-Aware Residual Stream**.

The **Statistical Trend Stream** serves as the backbone forecaster. Utilizing **TimeMixer**, it captures high-energy, inertial temporal patterns and global evolution, prioritizing stability to maintain robust baseline performance in noisy environments. By absorbing dominant trends, it enables the secondary stream to focus exclusively on modeling complex local deviations that standard regressors often miss.

The **Physics-Aware Residual Stream** captures transient fluctuations and regime shifts through two parallel branches. The **Static Branch** encodes time-invariant spatial constraints by fusing a domain-specific physical prior matrix ($\mathbf{A}^{\text{prior}}$) with learnable node embeddings, constructing a stable graph topology that respects fundamental system connectivity. Simultaneously, the **Dynamic Branch** addresses non-stationary physics via two key mechanisms:

1. An **Adaptive Window Mechanism** that learns flow-dependent transport delays ($\tau_{t,c}$). Unlike fixed lookback windows, this module dynamically adjusts the receptive field for each variable based on current operating conditions, aligning asynchronous signals caused by varying flow rates.

2. A **Physics-Guided Dynamic Graph** that separates causal interactions from spurious correlations by computing a time-varying adjacency matrix to capture transient couplings emerging only under specific regimes (e.g., high-load vs. idle states).

Finally, outputs from both streams are integrated via a **Gated Fusion Mechanism**. A learnable gating vector adaptively weights the physical residual contribution, adding it to the trend forecast only when regime-specific corrections are necessary. This architectural bias ensures adherence to physical laws without sacrificing statistical precision.




## Installation

The environment setup follows the standard `Time-Series-Library` benchmark but excludes heavy dependencies required for Large Language Models (LLMs).

**Requirements:**

* Python 3.8+
* PyTorch 1.10+
* NVIDIA CUDA toolkit (for GPU acceleration)

**Step 1: Clone the repository**

```bash
git clone https://github.com/YourUsername/DSPR-Industrial-Forecasting.git
cd DSPR-Industrial-Forecasting

```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt

```

*Recommended `requirements.txt` content:*

```text
torch>=1.10.0
numpy
pandas
scikit-learn
matplotlib
einops
tqdm
scipy

```

## Datasets

Due to licensing constraints, raw data is not distributed with this repo. Please download and place them in the `./dataset` directory.

1. **TEP (Tennessee Eastman Process)**:
* Download `TEP_FaultFree_Training.RData` from [Harvard Dataverse](https://www.google.com/search?q=https://dataverse.harvard.edu/dataset.xhtml%3FpersistentId%3Ddoi:10.7910/DVN/6C3JR1).


2. **SDWPF (Solar/Wind Power)**:
* Download `wtbdata_245days.csv` from [Baidu AI Studio](https://www.google.com/search?q=https://aistudio.baidu.com/datasetdetail/105634).


3. **SCR & Rotary Kiln**:
* These datasets are proprietary industrial data and cannot be released due to company privacy policies.



## Experiments & Results

We evaluate DSPR on four industrial benchmarks covering Chemical Kinetics (SCR), Thermodynamics (Kiln), Process Control (TEP), and Fluid Dynamics (SDWPF).

DSPR achieves Pareto-optimal performance, significantly reducing forecasting error (MAE/RMSE) while maintaining near-ideal physical consistency (MCA > 99%).

### Main Results (Normalized)

| Dataset | Model | MAE | RMSE | MCA (Conservation) | TVR (Fidelity) | TDA (Trend) |
| --- | --- | --- | --- | --- | --- | --- |
| **SCR** | **DSPR** | **0.265** | **0.415** | **99.8%** | **97.2%** | **83.5%** |
|  | TimeMixer | 0.286 | 0.435 | 99.1% | 88.5% | 74.9% |
|  | PatchTST | 0.287 | 0.442 | 97.9% | 91.2% | 78.6% |
| **Kiln** | **DSPR** | **0.291** | **0.436** | **99.5%** | **96.8%** | **81.0%** |
|  | TimeMixer | 0.308 | 0.465 | 98.8% | 84.2% | 72.5% |
| **TEP** | **DSPR** | **0.436** | **0.564** | **99.8%** | **95.4%** | **85.2%** |
|  | TimeMixer | 0.456 | 0.592 | 98.8% | 84.4% | 81.0% |
| **SDWPF** | **DSPR** | **0.335** | **0.522** | **99.2%** | **88.2%** | **-** |

* **MCA**: Mean Conservation Accuracy (Higher is better)
* **TVR**: Total Variation Ratio (Closer to 100% is better; baselines often over-smooth)
* **TDA**: Trend Directional Accuracy (Higher is better)

*(Full results comparison available in the paper).*

## 🚀 Usage

**Training Example**
To train the model on the TEP dataset:

```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/TEP/ \
  --data_path TEP.csv \
  --model_id TEP_96_96 \
  --model DSPR \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 52 \
  --dec_in 52 \
  --c_out 52 \
  --des 'Exp' \
  --itr 1



---

### Contact

For any questions, please open an issue or contact [yerazhang2-c@my.cityu.edu.hk](mailto:yerazhang2-c@my.cityu.edu.hk).
