---
language:
- pt
- en
license: mit
library_name: xgboost
tags:
- tabular-regression
- tabular-classification
- physics-informed-ml
- surrogate-model
- aerospace
- multidisciplinary-design-optimization
- sae-aerodesign
metrics:
- r2
- rmse
- mape
- roc_auc
- f1
---

# AeroMDO-Surrogate-XGB: Multidisciplinary Aircraft Design Surrogate Suite

## Model Summary
**AeroMDO-Surrogate-XGB** is an ensemble of gradient-boosted decision trees (`XGBRegressor` and `XGBClassifier`) trained to approximate coupled aerodynamic, longitudinal stability, and takeoff performance simulations for SAE Brasil AeroDesign aircraft (Monoplane and Biplane configurations).

## Dataset & Leak-Free Training Protocol
- **Total Raw Evaluations**: 49,500 historical MDO runs across 20 experimental campaigns (AVL Vortex Lattice and Lifting-Line Theory solvers; Particle Swarm Optimization and Pattern Search optimizers).
- **Cleaned Deduplicated Designs**: 37,309 unique configurations (29,847 train / 7,462 test).
- **Strictly Raw Input Features (16)**: `is_biplane`, `x_cg`, `b_asa`, `z_asa`, `gap`, `afil_asa`, `cr_asa`, `pr_asa`, `inc_asa`, `b_sh`, `x_sh_ba`, `z_sh`, `afil_sh`, `cr_sh`, `pr_sh`, `inc_sh`. Derived constraint variables (`ar_asa`, `ar_sh`, `s_asa`, `s_sh`, `lht`) are **excluded** from model features to eliminate feature leakage into the feasibility classifier.
- **Validation Protocols**:
  1. **Deduplicated 80/20 Stratified Split**: Measures within-domain interpolation across the 37,309 unique geometries.
  2. **5-Fold `GroupKFold` by `source_file` (18 distinct campaign files)**: Measures out-of-campaign generalization across unseen optimizer/solver trajectories.
  3. **Real AVL Vortex-Lattice Subset (`solver == 'AVL'`, 4,839 unique geometries)**: Measures accuracy strictly on high-fidelity 3D Vortex-Lattice runs from the historical spreadsheets.

## Evaluation Metrics

### 1. Aerodynamic Polar Validation (Table 4.3 — Relatório 2026, Grouped by Wing Geometry ID)
- **Split**: 480 training wings / 120 unseen test wings (5 angles of attack per wing):
  - $C_L(\alpha)$: $R^2 = 0.99952$, $\text{RMSE} = 0.00926$, $\text{MAPE} = 0.597\%$ (Target: $R^2 > 0.992$, $\text{MAPE} < 1.2\%$)
  - $C_D(C_L)$: $R^2 = 0.99985$, $\text{RMSE} = 0.00071$, $\text{MAPE} = 0.373\%$ (Target: $R^2 > 0.985$, $\text{MAPE} < 2.1\%$)

### 2. Real AVL Solver Subset Validation ($N_{\text{unique}} = 4,839$, Test $N = 968$)
- Pitching Moment $C_{m0}$: $R^2 = 0.9588$, $\text{RMSE} = 0.0529$
- Pitch Stiffness $C_{m\alpha}$: $R^2 = 0.9399$, $\text{RMSE} = 0.00385$
- Static Margin ($ME$): $R^2 = 0.9387$, $\text{RMSE} = 0.0475$
- Max Lift in Flight ($C_{L,\max}$): $R^2 = 0.8954$, $\text{RMSE} = 0.0683$, $\text{MAPE} = 1.60\%$
- Maximum Takeoff Weight ($MTOW$): $R^2 = 0.8377$, $\text{RMSE} = 0.7376\text{ kg}$, $\text{MAPE} = 3.64\%$

### 3. Global MDO Dataset: Deduplicated Split vs. 5-Fold `GroupKFold` by Campaign File
| Target / Task | Deduplicated 80/20 Split ($N=7,462$) | 5-Fold `GroupKFold` by `source_file` |
|---|---|---|
| **Constraint Feasibility Classifier (16 raw inputs)** | $\text{ROC-AUC} = 0.9898$, $F_1 = 0.9418$ | $\text{ROC-AUC} = 0.8661$, $F_1 = 0.5994$ ($\text{Prec} = 0.8834$) |
| **Pitch Stiffness ($C_{m\alpha}$)** | $R^2 = 0.9719$, $\text{RMSE} = 0.00287$ | $R^2 = 0.8948$, $\text{RMSE} = 0.00538$ |
| **Pitching Moment ($C_{m0}$)** | $R^2 = 0.9707$, $\text{RMSE} = 0.0398$ | $R^2 = 0.7736$, $\text{RMSE} = 0.1070$ |
| **Static Margin ($ME$)** | $R^2 = 0.9626$, $\text{RMSE} = 0.0352$ | $R^2 = 0.8550$, $\text{RMSE} = 0.0675$ |
| **Max Lift ($C_{L,\max}$ Flight)** | $R^2 = 0.9399$, $\text{MAPE} = 0.615\%$ | $R^2 = 0.6225$, $\text{MAPE} = 1.886\%$ |
| **Maximum Takeoff Weight ($MTOW$)** | $R^2 = 0.8919$, $\text{MAPE} = 1.980\%$ | $R^2 = 0.5812$, $\text{MAPE} = 4.197\%$ |
| **Flight Score ($PVOO$)** | $R^2 = 0.9594$, $\text{MAPE} = 2.702\%$ | Out-of-domain shift across Mono/Biplane campaigns |

### 4. Downstream Multi-Seed PSO Benchmark (5 Random Seeds: `42, 101, 202, 303, 404`)
- **Standard Full-Physics PSO (200 physics calls/run)**: $PVOO = 156.29 \pm 0.55$ ($36.55\text{ s}$)
- **Multi-Fidelity Surrogate PSO (40 physics calls/run, -80.0%)**: $PVOO = 153.05 \pm 1.15$ (**4.43x speedup**, **97.9% score retention**)
- **Equal-Budget Full-Physics PSO (40 physics calls/run)**: $PVOO = 153.97 \pm 1.26$ (statistically equivalent within $1\sigma$ due to historical monoplane span domain shift $b_w \ge 2.8\text{ m}$ vs. unconstrained search down to $b_w = 2.0\text{ m}$)
