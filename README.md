# TD-MPC2 Plasticity Study

This repository contains an experimental implementation of **TD-MPC2**, adapted from a TD-MPC codebase, together with tools to analyze **neural plasticity in model-based reinforcement learning**.

The project investigates how architectural choices introduced in TD-MPC2 — such as **SimNorm, Layer Normalization and Mish activations** — may influence the ability of the network to **maintain plastic representations during training**.

## Motivation

Temporal Difference Model Predictive Control (**TD-MPC**) is a model-based reinforcement learning algorithm that learns:

- a latent environment model  
- a value function  
- a policy  

while performing **short-horizon planning in latent space**.

<p align="center">
    <img src="media/ae591483.png" width="700"/>
    <br>
    <i>Schematic representation of the path-planning and path-following pipeline.. </i>
</p>

TD-MPC2 introduces several architectural improvements designed to improve **scalability and robustness across tasks**.

A key hypothesis explored in this project is that these improvements may partly stem from **better preservation of neural plasticity**.

In reinforcement learning, maintaining plasticity is particularly important because the training distribution is **non-stationary**: policy updates modify the data distribution, which may cause issues such as

- gradient collapse
- feature rank collapse
- representation saturation.

This repository studies how TD-MPC2 components may mitigate such effects.

--- 

# Implemented TD-MPC2 Components

The table below summarizes which components of the original **TD-MPC2 architecture** are currently implemented.

| Component | TD-MPC | TD-MPC2 | Implemented |
|-----------|--------|---------|-------------|
| Network architecture | MLP, ELU, no LayerNorm | MLP + LayerNorm + Mish + SimNorm on latent state | Yes |
| Latent regularization | None | SimNorm | Yes |
| Q-functions | 2 Q-functions | Ensemble of 5 Q-functions + 1% Dropout | Yes |
| TD target | Standard double-Q | Min over randomly sampled Q-functions | Yes |
| Model size | ~1M parameters | ~5M parameters | No |
| Policy prior | Deterministic + Gaussian noise | Maximum entropy RL | Yes |
| Planning | MPPI with momentum | MPPI without momentum | Yes |
| Model objective | Continuous regression | Discrete regression in log space | No |
| Reward robustness | Sensitive to reward scale | Scale-invariant loss | No |
| Multi-task learning | Limited | Task-conditioned world models | No |
| Replay buffer | Prioritized replay | Uniform sampling + multi-worker | No |
| Implementation optimizations | Standard | Q-ensemble vectorization | No |

---

# Plasticity Metrics

Several metrics are used to analyze the evolution of network plasticity during training:

- **Weight magnitude**
- **Weight distance**
- **Gradient norm**
- **Zero Gradient Ratio (ZGR)**
- **Feature Zero Activation Ratio (FZAR)**
- **Stable rank of features**
- **Unnormalized Neural Tangent Kernel (ENTK)**

These metrics help characterize:

- representation sparsity
- gradient flow
- effective feature dimensionality
- adaptation capacity.

---

# Example Results

Example training diagnostics:

--- 
# Related Repository

This project relies on a second repository used to instrument **TD-MPC with plasticity diagnostics**: [https://github.com/lauraachoquett/tdmpc_plasticity]

➡ **TD-MPC Plasticity Instrumentation**

That repository contains:

- the baseline TD-MPC implementation
- incremental architectural modifications
- full plasticity metrics.
