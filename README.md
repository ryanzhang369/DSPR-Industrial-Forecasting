# DSPR-Industrial-Forecasting

This repository contains the official PyTorch implementation for the paper:

**"DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting"**.

## Introduction

Forecasting complex industrial systems—spanning chemical kinetics, thermal dynamics, and energy meteorology—requires balancing statistical precision with physical plausibility. While standard deep learning models often achieve low prediction errors (MSE/MAE), they frequently violate fundamental conservation laws and causal relationships, a phenomenon we term **"fidelity collapse."**

**DSPR (Dual-Stream Physics-Residual Network)** addresses this challenge by shifting physics integration from passive loss penalties to **active architectural inductive biases**. Rather than forcing a single model to capture all dynamics, DSPR explicitly decouples the forecasting workflow into two specialized streams:

1. **Statistical Trend Stream**: Captures high-energy, inertial temporal patterns using a robust statistical forecaster, ensuring stable baseline performance.

2. **Physics-Aware Residual Stream**: Models regime-dependent deviations and transient dynamics through physics-guided dynamic graphs and adaptive temporal windows that respect flow-dependent transport delays.

This architectural decoupling enables DSPR to achieve state-of-the-art predictive accuracy while maintaining near-ideal physical fidelity, bridging the gap between data-driven forecasting and trustworthy industrial deployment.


## Methodology

![DSPR Architecture](figures/fig_architecture.jpg)
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
git clone https://github.com/ryanzhang369/DSPR-Industrial-Forecasting.git
cd DSPR-Industrial-Forecasting

```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt

```


## Datasets

Due to licensing constraints, raw data is not distributed with this repo. Please download and place them in the `./dataset` directory.

1. **TEP (Tennessee Eastman Process)**:
* Download `TEP_FaultFree_Training.RData` from [[Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1#)].

2. **SDWPF (Solar/Wind Power)**:
* Download `wtbdata_245days.csv` from [[Baidu AI Studio](https://aistudio.baidu.com/competition/detail/152/0/introduction)].

3. **SCR & Rotary Kiln**:
* These datasets are proprietary industrial data and cannot be released due to company privacy policies.



## Experiments & Full Results

> **Note:** Due to strict space constraints in the KDD 2026 proceedings, the granular performance breakdown across varying prediction horizons is provided here as supplementary material. For offline reading, please refer to the [`Supplementary_Material.pdf`](./Supplementary_Material.pdf) located in the repository root.

We evaluate DSPR on four diverse industrial datasets spanning Chemical Kinetics (**SCR**), Thermodynamics (**Rotary Kiln**), Process Control (**TEP**), and Fluid Dynamics (**SDWPF**). These benchmarks represent a spectrum from micro-scale reactions to macro-scale environmental physics, rigorously testing DSPR's generalization across heterogeneous physical regimes.

### Physical Time Constants & Evaluation Horizons (H)

A key distinction of our evaluation protocol is that the prediction horizons (*H*) are not uniform across all datasets. Instead, they are explicitly customized to align with the intrinsic physical time constants and control dynamics of each system:

* **SCR (Chemical Kinetics):** We select short-to-medium horizons (H=24, 48, 96, 192) to capture the rapid chemical reaction kinetics and variable transport delays (seconds to minutes) characteristic of denitrification processes.
* **Kiln (Thermodynamics):** Given the massive thermal inertia of the rotary kiln, we extend horizons (H=96, 192, 336, 720) to cover longer durations, enabling the assessment of slow-moving thermodynamic trends and combustion efficiency shifts.
* **TEP (Process Control):** Horizons are restricted to the *transient response window* (H=6, 12, 18, 24). This range effectively covers the open-loop dynamic phase before feedback controllers fully stabilize the reactor pressure, avoiding the trivial task of predicting steady-state setpoints.
* **SDWPF (Wind Energy):** In the absence of Numerical Weather Predictions (NWP), we limit evaluation to the *inertial forecasting regime* (H=12, 24, 36, 48). This strictly targets the ultra-short-term dispatch market, where local kinematic history retains predictive validity before atmospheric chaos dominates.

### Full Performance Breakdown

DSPR achieves Pareto-optimal performance across all benchmarks. It simultaneously reduces forecasting statistical errors (MAE/RMSE) compared to state-of-the-art baselines while enforcing strict adherence to physical laws—successfully resolving the accuracy-fidelity dilemma that typically plagues conventional data-driven models.

![Granular Performance Breakdown](figures/full_results.jpg)
*Figure 3: **Granular performance comparison on industrial benchmarks.** DSPR not only achieves the lowest MAE/RMSE but also maintains **>99% Mean Conservation Accuracy** and high signal fidelity (**TVR 83%–97%**) across both short-term transients and long-term horizons.*

### Key Evaluation Metrics

To rigorously evaluate Physical Consistency alongside statistical accuracy, we utilize three specialized metrics:

* **MCA (Mean Conservation Accuracy):** Quantifies the percentage of predictions satisfying physical constraints (e.g., mass/energy balance) relative to the ground truth. Higher values indicate better physical consistency.
* **TVR (Total Variation Ratio):** Assesses whether the model captures realistic signal volatility versus producing over-smoothing artifacts. Values approaching 100% indicate the successful preservation of physically meaningful high-frequency transients.
* **TDA (Trend Directional Accuracy):** Evaluates the correctness of predicted trend directions during significant state shifts, measuring the model's adherence to physical causality and its ability to anticipate regime transitions.



## Usage

**Training Example**
To train the model on the TEP dataset:

```bash
python main_dspr.py \
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
```

---

### Contact

For any questions, please open an issue or contact [yerazhang2-c@my.cityu.edu.hk](mailto:yerazhang2-c@my.cityu.edu.hk).
