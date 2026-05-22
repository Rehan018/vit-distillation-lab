# Knowledge Distillation Experiments Results

## Teacher Ceiling

| Model | Params | Pets Acc |
|---|---|---|
| Teacher (ViT-Large) | 307M | 91.90% |
| Student Baseline (ViT-Tiny) | 5.7M | 53.80% |
| Teacher–Student Gap | — | 38.10% |

## Isolated (Deconfounded) Experiments — Single Seed

| Experiment | Config File | Pets Acc | Gain | Gap Recovery |
|---|---|---|---|---|
| Baseline | `baseline.yaml` | 53.80% | — | 0% |
| Vanilla KD | `kd_baseline.yaml` | 64.40% | +10.60% | 27.8% |
| Feature KD (λ=0.1) | `feature_kd_lam0_1.yaml` | 69.50% | +15.70% | 41.2% |
| Attention KD Only | `attention_kd_only.yaml` | 68.10% | +14.30% | 37.5% |
| Relational KD Only | `relational_kd_only.yaml` | 67.90% | +14.10% | 37.0% |

## Multi-Seed Variance Analysis (3 Seeds: 42, 123, 456)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---|---|---|---|---|
| Baseline (No KD) | 57.10% | 41.60% | 54.80% | **51.17 ± 6.83%** |
| Vanilla KD | 65.80% | 66.70% | 64.60% | **65.70 ± 0.86%** |
| Feature KD (λ=0.1) | 71.20% | 69.00% | 65.00% | **68.40 ± 2.57%** |
| Attention KD Only | 68.30% | 66.20% | 58.80% | **64.43 ± 4.07%** |
| Relational KD Only | 68.00% | 66.50% | 60.50% | **65.00 ± 3.24%** |

## Feature KD λ Ablation

| Experiment | Config File | Pets Acc | Gain |
|---|---|---|---|
| Feature KD (λ=1.0) | `feature_kd_lam1_0.yaml` | 63.40% | +9.60% |
| Feature KD (λ=0.1) | `feature_kd_lam0_1.yaml` | 69.50% | +15.70% |
| Feature KD (λ=0.01) | `feature_kd_lam0_01.yaml` | 64.40% | +10.60% |

## Combined (Multi-Objective) Experiments

| Experiment | Config File | Pets Acc | Gain | Note |
|---|---|---|---|---|
| Attention KD + Feature KD | `attention_kd.yaml` | 63.30% | +9.50% | Feature KD λ=1.0 active |
| Relational KD + All | `relational_kd.yaml` | 65.20% | +11.40% | Feature + Attention KD active |
