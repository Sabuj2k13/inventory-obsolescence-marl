# SKU-Level Inventory Obsolescence Control
### Explainable Classification · Multi-Horizon Forecasting · Surrogate-Assisted MARL

> A unified predictive-prescriptive pipeline that moves from risk diagnosis to coordinated intervention — without relying on disconnected dashboards or static heuristics.

---

## Overview

Inventory obsolescence silently erodes working capital through elevated holding costs, markdown losses, and forced liquidations. Existing approaches treat it as a narrow forecasting or replenishment problem. This research reframes it as an **end-to-end intelligence problem** requiring four simultaneous answers:

| Question | Module |
|---|---|
| Which SKUs are already at risk? | Explainable Hybrid Classifier |
| How long will their inactivity persist? | Multi-Horizon MSLS Forecaster |
| What latent behavioural segments exist? | KMeans + DBSCAN Clustering |
| What intervention should be taken next? | Surrogate-Assisted MARL |

---

## Pipeline Architecture

```
Raw SKU Data (ProductCode · Month · INVCat · MSLS · Operational Features)
        │
        ▼
┌─────────────────────────────────────────┐
│  Task A — Explainable Classification    │
│  XGBoost · LightGBM · RF · BiLSTM      │
│  → Weighted Hybrid Ensemble             │
│  → SHAP explanations per SKU           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Task B — Leakage-Robust Forecasting    │
│  RF Regression · LSTM Regression        │
│  Horizons: H+3 · H+6 · H+11 (MSLS)    │
│  Correlation gate removes proxy leakage │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Task C — Behavioural Segmentation      │
│  KMeans (k=3) · DBSCAN anomaly flagging │
│  Features: velocity · capital lock ·    │
│  zero-sales behaviour · margin · PCR    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Task D — Surrogate Inventory Env       │
│  World model trained on state transitions│
│  Stochastic noise · demand jitter       │
│  Reward: ↓ holding cost · ↑ velocity   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Task E — MARL Controller               │
│  Agents: Monitor · Procurement ·        │
│           Liquidation                   │
│  Primary: QMIX                         │
│  Baselines: MAPPO · MADDPG ·           │
│             Rule-Based · No-Action      │
└─────────────────────────────────────────┘
```

---

## Results

### Task A — Risk Classification

Three-class stratification: **Regular · Obsolete · End-of-Life**

| Model | Accuracy | Macro-F1 | AUC-ROC |
|---|---|---|---|
| **XGBoost** | **85.71%** | **0.7897** | **0.9237** |
| LightGBM | 85.81% | 0.7897 | 0.9219 |
| Random Forest | 85.13% | 0.7793 | 0.9279 |
| BiLSTM | 66.54% | 0.6102 | 0.8408 |
| Hybrid Ensemble | 85.13% | 0.7880 | 0.9175 |

XGBoost achieves the strongest Macro-F1 (0.7897) with an AUC-ROC of 0.9237. SHAP-based feature selection identifies the most actionable risk signals. The Obsolete class is the hardest to separate (F1 ≈ 0.57), reflecting its naturally transitional nature.

---

### Task B — Multi-Horizon MSLS Forecasting

Predicting **Months Since Last Sale** without proxy inactivity leakage.

| Model | Horizon | MAE | RMSE | R² |
|---|---|---|---|---|
| **Random Forest** | **H+3** | **0.311** | 0.981 | **0.930** |
| **Random Forest** | **H+6** | **0.386** | 1.137 | **0.902** |
| **Random Forest** | **H+11** | **0.449** | 1.298 | **0.872** |
| LSTM | H+3 | 0.704 | 1.316 | 0.873 |
| LSTM | H+6 | 0.812 | 1.710 | 0.778 |
| LSTM | H+11 | 0.863 | 2.261 | 0.612 |

Random Forest dominates across all horizons. R² degrades gracefully with horizon length — confirming robust generalisation rather than proxy-inflated accuracy.

---

### Task C — Behavioural Segmentation

| Metric | Value |
|---|---|
| Silhouette Score | **0.7594** |
| Davies-Bouldin Index | 0.509 |
| Clusters (k) | 3 |
| Outliers flagged (DBSCAN) | 4 SKUs |

Features: `StockVelocity · ZeroSalesMonthsCount · CapitalLockRatio · HoldingCostEstimate · GrossMarginPct · PCR`

A silhouette score of 0.76 indicates well-separated, compact behavioural segments — enabling policy learning that is tailored to each risk tier.

---

### Task D — World Model (Surrogate Environment)

| Metric | Value |
|---|---|
| MAE | 0.00135 |
| R² | 0.9990 |
| Noise σ | 0.05 |
| State dimensions | 10 |

The world model explains 99.9% of observed state-transition variance, providing a reliable simulation substrate for policy training.

---

### Task E — MARL Controller Performance

Three cooperative agents — **Monitor · Procurement · Liquidation** — trained in the surrogate environment.

| Algorithm | Mean Reward | Obso. Reduction | HCE Reduction | Turnover | Action Diversity |
|---|---|---|---|---|---|
| **QMIX** *(primary)* | **1.854 ± 0.147** | **96.67%** | 1.56% | 604.58 | 1.717 |
| MAPPO | 1.837 ± 0.200 | 99.67% | 5.75% | 351.75 | 1.615 |
| MADDPG | 1.512 ± 0.569 | 90.00% | −18.22% | 263.87 | 1.476 |
| Rule-Based | −2.707 | 1.67% | — | 348.86 | — |
| No-Action | −3.516 | 0% | — | 702.81 | — |

**QMIX** is selected as the primary controller: cross-seed mean reward of +1.854 is statistically significant (p = 0.0004), no agents collapse, and policy entropy of 1.987 confirms diverse, non-degenerate behaviour.

**Dominant agent actions (QMIX):** Monitor → `F-Med` (46.8%) · Procurement → `Stop` (45.1%) · Liquidation → `Disc-20` (46.9%)

---

### Multi-Seed Stability

| Algorithm | Seeds | Mean Reward | Seed Std | p-value | Significant |
|---|---|---|---|---|---|
| QMIX | 42 · 123 · 999 | 1.854 | 0.147 | 0.00037 | ✓ |
| MAPPO | 42 · 123 · 999 | 1.837 | 0.200 | 0.0007 | ✓ (one seed collapses) |
| MADDPG | 42 · 123 · 999 | 1.512 | 0.569 | 0.006 | ✓ |

---

## Research Objectives

| ID | Objective |
|---|---|
| RO1 | Leakage-robust explainable classification into Regular / Obsolete / EOL |
| RO2 | Multi-horizon MSLS forecasting at H+3, H+6, H+11 |
| RO3 | Behavioural segmentation and anomaly detection |
| RO4 | Surrogate inventory environment from observed state transitions |
| RO5 | MARL controller training and benchmarking |
| RO6 | Predictive accuracy, policy stability, and ablation evaluation |
| RO7 | Deployable Streamlit dashboard for unseen CSV scoring |

---

## Repository Structure

```
├── results/
│   ├── task_a_metrics.csv              # Classification performance
│   ├── task_b_regression_metrics.csv   # Forecasting metrics (all horizons)
│   ├── task_c_cluster_metrics.csv      # Clustering evaluation
│   ├── marl_eval_results.csv           # Per-policy MARL results
│   ├── marl_algorithm_comparison.csv   # Full algorithm comparison table
│   ├── multiseed_stats_all_algos.csv   # Cross-seed stability analysis
│   ├── ablation_results.csv            # Component ablation
│   ├── world_model_metrics.csv         # Surrogate environment quality
│   ├── collapse_detection_report.csv   # Agent action-collapse diagnostics
│   ├── shap_importance.csv             # SHAP feature rankings
│   ├── final_sku_scored_output.csv     # Full SKU-level scored output
│   └── summary_results.csv            # Publication-ready summary
│
├── figures/
│   ├── task_a_model_comparison.pdf
│   ├── cm_*.pdf                        # Per-model confusion matrices
│   ├── roc_*.pdf                       # Per-model ROC curves
│   ├── shap_importance.pdf
│   ├── task_b_scatter_h*.pdf           # Forecast scatter plots
│   ├── task_c_cluster_scatter.pdf
│   ├── task_c_cluster_composition.pdf
│   ├── task_c_elbow_silhouette.pdf
│   ├── world_model_loss.pdf
│   ├── marl_reward_curves_seed*.pdf
│   ├── marl_vs_baseline.pdf
│   ├── ablation_comparison.pdf
│   ├── algo_learning_curves.pdf
│   ├── algo_action_distributions.pdf
│   ├── algo_policy_heatmaps.pdf
│   └── algo_bar_comparison.pdf
│
└── models/
    ├── lstm_clf.pt                     # BiLSTM classifier
    ├── lstm_reg_h3.pt                  # LSTM forecaster H+3
    ├── lstm_reg_h6.pt                  # LSTM forecaster H+6
    ├── lstm_reg_h11.pt                 # LSTM forecaster H+11
    ├── world_model.pt / world_model_v5.pt
    ├── qmix_agent_[0-2].pt
    ├── qmix_mixer.pt
    ├── mappo_actor_[0-2].pt
    └── mappo_critic.pt
```

---

## Key References

Teunter et al. (2011) · Babai et al. (2019) · Petropoulos et al. (2019) · Shajalal et al. (2022, 2023) · Ahmadi et al. (2022) · Park et al. (2023) · Mousa et al. (2024) · Kotecha & del Rio Chanona (2025) · Sagaert & Kourentzes (2025) · Jauhar et al. (2025)
