# Knowledge Distillation Experiments Results

## Teacher Ceiling

| Model | Params | Pets Acc | CIFAR-100 Acc |
|---|---|---|---|
| Teacher (ViT-Large) | 307M | 91.90% | 86.68% |
| Student Baseline (ViT-Tiny) | 5.7M | 53.80% | 35.20% |
| Teacher–Student Gap | — | 38.10% | 51.48% |

## Core Experiments — ViT-Tiny Student (Seed 42)

| # | Experiment | Config File | Pets Acc | CIFAR-100 | Gain (Pets) | Gap Recovery |
|---|---|---|---|---|---|---|
| E0 | Baseline | `baseline.yaml` | 53.80% | 35.20% | — | 0% |
| E1 | Vanilla KD | `kd_baseline.yaml` | 64.40% | 42.10% | +10.60% | 27.8% |
| E2 | Feature KD (λ=0.1) | `feature_kd_lam0_1.yaml` | 69.50% | 45.80% | +15.70% | 41.2% |
| E3 | Attention KD Only | `attention_kd_only.yaml` | 68.10% | 43.60% | +14.30% | 37.5% |
| E4 | Relational KD Only | `relational_kd_only.yaml` | 67.90% | 43.20% | +14.10% | 37.0% |
| E5 | DKD (Decoupled) | `dkd.yaml` | 66.80% | 44.50% | +13.00% | 34.1% |
| E6 | Combined Feat+Attn | `combined_feat_attn.yaml` | 68.90% | 44.20% | +15.10% | 39.6% |
| E7 | Feature KD (CLS-only) | `feature_kd_cls_only.yaml` | 67.20% | 44.00% | +13.40% | 35.2% |
| E8 | Feature KD (Patch-only) | `feature_kd_patch_only.yaml` | 65.40% | 40.80% | +11.60% | 30.4% |

## Student Model Scaling — ViT-Small (22M, 384D)

| Method | ViT-Tiny Pets | ViT-Small Pets | Δ |
|---|---|---|---|
| Baseline | 53.80% | 68.50% | +14.70% |
| Vanilla KD | 64.40% | 74.20% | +9.80% |
| Feature KD (λ=0.1) | 69.50% | 77.30% | +7.80% |
| Attention KD | 68.10% | 76.80% | +8.70% |
| Relational KD | 67.90% | 75.60% | +7.70% |

## Multi-Seed Variance Analysis (3 Seeds: 42, 123, 456)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---|---|---|---|---|
| Baseline (No KD) | 57.10% | 41.60% | 54.80% | **51.17 ± 8.36%** |
| Vanilla KD | 65.80% | 66.70% | 64.60% | **65.70 ± 1.05%** |
| Feature KD (λ=0.1) | 71.20% | 69.00% | 65.00% | **68.40 ± 3.14%** |
| Attention KD Only | 68.30% | 66.20% | 58.80% | **64.43 ± 4.99%** |
| Relational KD Only | 68.00% | 66.50% | 60.50% | **65.00 ± 3.97%** |

## Feature KD λ Ablation

| Experiment | Config File | Pets Acc | Gain |
|---|---|---|---|
| Feature KD (λ=1.0) | `feature_kd_lam1_0.yaml` | 63.40% | +9.60% |
| Feature KD (λ=0.1) | `feature_kd_lam0_1.yaml` | 69.50% | +15.70% |
| Feature KD (λ=0.01) | `feature_kd_lam0_01.yaml` | 64.40% | +10.60% |

## Combined (Multi-Objective) Experiments — Legacy

| Experiment | Config File | Pets Acc | Gain | Note |
|---|---|---|---|---|
| Attention KD + Feature KD | `attention_kd.yaml` | 63.30% | +9.50% | Feature KD λ=1.0 active (confounded) |
| Relational KD + All | `relational_kd.yaml` | 65.20% | +11.40% | Feature + Attention KD active (confounded) |
