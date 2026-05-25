# Knowledge Distillation Experiment Results

This file reports only metrics that are backed by artifacts in the current checkout. CIFAR-100 uses a 1k subset with 3 probe epochs; NYUv2 and VOC use 80 train / 80 validation samples with 2 probe epochs.

## Artifact-Backed Results

| Model / Method | Config | Oxford Pets Acc | Gain vs Tiny baseline | CIFAR-100 Acc | NYUv2 RMSE / δ1 | VOC mIoU |
|---|---|---:|---:|---:|---:|---:|
| Teacher ViT-Large | `evaluate_teacher.py` | 91.90% | +38.10% | 79.90% | 2.9877 / 2.67% | 15.11% |
| Student baseline ViT-Tiny | `baseline.yaml` | 53.80% | +0.00% | 15.80% | 2.9953 / 0.00% | 6.09% |
| Vanilla KD | `kd_baseline.yaml` | 64.40% | +10.60% | 18.80% | 2.8029 / 0.70% | 8.95% |
| Feature KD, λ=0.1 | `feature_kd_lam0_1.yaml` | **69.50%** | **+15.70%** | 16.90% | 1.7741 / 27.36% | 6.99% |
| Attention KD only | `attention_kd_only.yaml` | 68.10% | +14.30% | 22.30% | 2.5888 / 2.93% | 10.46% |
| Relational KD only | `relational_kd_only.yaml` | 67.90% | +14.10% | 22.20% | 2.5759 / 3.19% | 10.63% |
| DKD | `dkd.yaml` | **79.10%** | **+25.30%** | **29.30%** | 3.0818 / 0.00% | **11.05%** |
| Combined Feature+Attention KD | `combined_feat_attn.yaml` | 75.60% | +21.80% | 18.30% | 2.0194 / 18.39% | 6.00% |
| Feature KD, CLS-only | `feature_kd_cls_only.yaml` | 75.20% | +21.40% | 22.60% | 2.8533 / 0.33% | 10.02% |
| Feature KD, Patch-only | `feature_kd_patch_only.yaml` | 74.90% | +21.10% | 18.00% | **1.6067 / 33.55%** | 6.00% |
| Student baseline ViT-Small | `baseline_small.yaml` | 83.60% | +29.80% | 42.00% | 1.7369 / 25.55% | 11.06% |
| Feature KD ViT-Small | `feature_kd_small.yaml` | 87.00% | +33.20% | 48.10% | 2.1021 / 13.94% | 13.91% |
| Legacy Attention+Feature KD | `attention_kd.yaml` | 63.30% | +9.50% | No dense/classification artifact beyond Pets | No artifact | No artifact |
| Legacy Relational+Feature+Attention KD | `relational_kd.yaml` | 65.20% | +11.40% | No dense/classification artifact beyond Pets | No artifact | No artifact |

## Feature KD λ Ablation

| Experiment | Config File | Oxford Pets Acc | Gain vs baseline |
|---|---|---:|---:|
| Feature KD, λ=1.0 | `feature_kd_lam1_0.yaml` | 63.40% | +9.60% |
| Feature KD, λ=0.1 | `feature_kd_lam0_1.yaml` | **69.50%** | **+15.70%** |
| Feature KD, λ=0.01 | `feature_kd_lam0_01.yaml` | 64.40% | +10.60% |

## Multi-Seed Variance Analysis

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---|---:|---:|---:|---:|
| Baseline (No KD) | 57.10% | 41.60% | 54.80% | **51.17 ± 8.36%** |
| Vanilla KD | 65.80% | 66.70% | 64.60% | **65.70 ± 1.05%** |
| Feature KD, λ=0.1 | 71.20% | 69.00% | 65.00% | **68.40 ± 3.14%** |
| Attention KD only | 68.30% | 66.20% | 58.80% | **64.43 ± 4.99%** |
| Relational KD only | 68.00% | 66.50% | 60.50% | **65.00 ± 3.97%** |

## Additional Implemented Runs Still Missing Current Artifacts

| Experiment axis | Configs / scripts | Status |
|---|---|---|
| ViT-Small scaling variants | `configs/vanilla_kd_small.yaml`, `configs/attention_kd_small.yaml`, `configs/relational_kd_small.yaml` | Implemented; no saved artifact in this checkout |
| CIFAR-100 | `evaluate_cifar100.py` | Complete for selected matrix |
| NYUv2 depth | `evaluate_depth.py` | Complete for selected matrix |
| Pascal VOC segmentation | `evaluate_segmentation.py` | Complete for selected matrix |

## Best Improvements

The best ViT-Tiny classification result is DKD: 79.10% on Pets (+25.30) and 29.30% on CIFAR-100 (+13.50). The best ViT-Tiny NYUv2 depth result is Patch-only Feature KD at RMSE 1.6067 / δ1 33.55. The best ViT-Tiny VOC result is DKD at 11.05 mIoU, with Relational KD (10.63) and Attention KD (10.46) close behind. The best overall student classification result is ViT-Small Feature KD at 87.00% Pets and 48.10% CIFAR-100.
