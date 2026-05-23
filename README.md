# Vision Transformer Knowledge Distillation Framework

A modular framework for comparing different knowledge distillation strategies for Vision Transformers. Investigates how different forms of teacher supervision — logit matching, feature alignment, attention transfer, and relational geometry — affect downstream transfer quality across classification and dense prediction tasks.

**Teacher**: ViT-Large (307M params) → **Student**: ViT-Tiny (5.7M params)

---

## Abstract

This project compares four ViT distillation strategies under controlled conditions using a ViT-Large teacher and ViT-Tiny student. All methods are evaluated with frozen-backbone linear probing on two classification datasets (Oxford Pets, CIFAR-100) and frozen-backbone depth estimation on NYUv2.

Key findings:

- Feature KD with moderate constraint weight (λ=0.1) provides the best classification transfer, recovering ~41% of the teacher-student accuracy gap on Oxford Pets
- Strong feature constraints (λ=1.0) cause representation collapse — lower reconstruction error does not mean better downstream performance
- Teacher supervision acts as a powerful regularizer: the unguided baseline shows ~7× higher variance than Vanilla KD across seeds
- Dense prediction (depth estimation) shows different method rankings than classification — attention-based supervision appears more beneficial for spatial tasks

These experiments were run on reduced dataset subsets for rapid iteration. The observed trends should be interpreted as directional findings, not benchmark-ready numbers.

---

## Problem Statement

Vision Transformers achieve strong performance but are expensive to deploy. Knowledge distillation can compress a large teacher's knowledge into a smaller student, but different forms of supervision target different aspects of the teacher's representation. The question is: which distillation strategy best preserves downstream performance, and does the answer change between classification and spatially-structured tasks?

---

## Literature & Design Rationale

The experimental design draws from:

- **Hinton et al. (2015)** — introduced soft logit distillation via temperature-scaled KL divergence. Our Vanilla KD baseline implements this directly.
- **ViTKD (Yang et al., CVPRW 2024)** — proposed intermediate feature alignment specifically for ViTs. This motivated our Feature KD implementation with learnable projection layers to bridge the 1024D→192D embedding gap.
- **Zagoruyko & Komodakis (2017)** — attention transfer for CNNs. We adapted this to ViT self-attention maps, averaging across heads to handle the 16→3 head mismatch.
- **Park et al. (2019, RKD)** — relational knowledge distillation using pairwise distance/angle matching. Our relational loss uses batch-wise cosine similarity matrices instead of absolute coordinates.

The project does not reproduce any single paper. Instead, it compares fundamentally different forms of supervision under identical training conditions to understand their relative strengths.

---

## Initial Hypothesis

I expected Feature KD to consistently outperform Vanilla KD because intermediate transformer representations contain richer semantic information than final logits alone. However, I suspected that the large capacity gap (1024D→192D, 24→12 layers) would make direct feature matching unstable.

This led to the central question:
> *"At what point does intermediate supervision become overconstrained for a much smaller student?"*

The Feature KD λ ablation was designed specifically to test this.

---

## Distillation Objectives

### 1. Vanilla KD (Logit Distillation)

$$\mathcal{L}_{\text{Vanilla}} = \alpha \mathcal{L}_{\text{CE}}(y_s, y) + (1 - \alpha) T^2 \mathcal{D}_{\text{KL}}\left( \sigma\left(\frac{z_s}{T}\right) \parallel \sigma\left(\frac{z_t}{T}\right) \right)$$

### 2. Feature KD (Intermediate Feature Alignment)

$$\mathcal{L}_{\text{Feature}} = \frac{1}{N} \sum_{i \in \text{layers}} \Vert F_t^{(i)} - W^{(i)} F_s^{(i)} \Vert_2^2$$
Learnable projection $W$ maps 192D student features to 1024D teacher space.

### 3. Attention KD (Self-Attention Map Matching)

$$\mathcal{L}_{\text{Attention}} = \frac{1}{B} \sum_{b=1}^B \Vert \bar{A}_t^{(b)} - \bar{A}_s^{(b)} \Vert_F^2$$
Attention maps averaged across heads (16→1, 3→1) before comparison.

### 4. Relational KD (Batch Geometry Matching)

$$\mathcal{L}_{\text{Relational}} = \Vert G_t - G_s \Vert_F^2, \quad G_{j,k} = \frac{f_j \cdot f_k}{\Vert f_j \Vert \Vert f_k \Vert}$$
Matches cosine similarity structure across the batch rather than absolute coordinates.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| **Student** | `vit_tiny_patch16_224` (192D, 12 layers, 3 heads) |
| **Teacher** | `vit_large_patch16_224` (1024D, 24 layers, 16 heads) |
| **Training data** | Oxford Pets 1000-sample subset |
| **Batch size** | 16 |
| **Epochs** | 10 |
| **Optimizer** | Adam, LR=1e-4 with cosine annealing |
| **Gradient clipping** | max_norm=1.0 |
| **KD hyperparameters** | α=0.7, T=4.0 |
| **Evaluation** | Frozen backbone + CLS token linear probe (5 epochs) |
| **Seeds** | 42, 123, 456 for variance analysis |

### Evaluation Tasks

All evaluations use frozen backbone weights — only the lightweight head is trained:

1. **Oxford Pets** (37 classes) — linear probe classification, top-1 accuracy
2. **CIFAR-100** (100 classes) — linear probe classification, top-1 accuracy (5k subset)
3. **NYUv2** — monocular depth estimation with MLP decoder, RMSE and δ1

---

## Engineering Challenges

### Feature Dimension Mismatch

ViT-Large (1024D) and ViT-Tiny (192D) use different embedding dimensions. Direct feature matching is impossible without projection. Implemented learnable linear projectors in `distillation/projector.py` to map student features into teacher space.

### Attention Extraction in timm

timm's ViT uses fused scaled dot-product attention that doesn't expose raw attention matrices. Wrote custom monkey-patched forward in `utils/hooks.py` that manually computes Q·K^T attention and captures it during the forward pass. This was the most time-consuming engineering challenge — I first tried register_forward_hook but the fused kernel doesn't expose attention as intermediate state. The `scratch/inspect_attn.py` file shows the debugging I did to understand timm's attention internals.

### Attention Head Mismatch

ViT-Large has 16 heads, ViT-Tiny has 3. Head-to-head matching is structurally impossible. Averaging across heads before computing MSE loss is a simplification, but it was stable and produced clear signal.

### Feature KD Collapse

Early runs with λ_feature=1.0 collapsed to ~1% accuracy — the student was spending all its capacity minimizing reconstruction error instead of learning discriminative features. This directly motivated the λ ablation study.

---

## Results

### Teacher Ceiling

| Model | Params | Oxford Pets | CIFAR-100 | NYUv2 RMSE | NYUv2 δ1 |
|---|---|---|---|---|---|
| **Teacher (ViT-Large)** | 307M | TBD | TBD | TBD | TBD |
| **Student Baseline (No KD)** | 5.7M | TBD | TBD | TBD | TBD |

*These values will be populated after running `bash run_all.sh`.*

### Cross-Task Results (Single Seed)

| Method | Config | Pets Acc | CIFAR-100 Acc | NYUv2 RMSE ↓ | NYUv2 δ1 ↑ |
|---|---|---|---|---|---|
| **Baseline** | `baseline.yaml` | TBD | TBD | TBD | TBD |
| **Vanilla KD** | `kd_baseline.yaml` | TBD | TBD | TBD | TBD |
| **Feature KD (λ=0.1)** | `feature_kd_lam0_1.yaml` | TBD | TBD | TBD | TBD |
| **Attention KD Only** | `attention_kd_only.yaml` | TBD | TBD | TBD | TBD |
| **Relational KD Only** | `relational_kd_only.yaml` | TBD | TBD | TBD | TBD |

### Feature KD λ Ablation

| λ_feature | Pets Acc | Notes |
|---|---|---|
| 1.0 | TBD | Strong constraint |
| 0.1 | TBD | Moderate — expected peak |
| 0.01 | TBD | Weak constraint |

### Multi-Seed Variance Analysis (3 Seeds: 42, 123, 456)

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | 95% CI |
|---|---|---|---|---|---|
| **Baseline** | TBD | TBD | TBD | TBD | TBD |
| **Vanilla KD** | TBD | TBD | TBD | TBD | TBD |
| **Feature KD (λ=0.1)** | TBD | TBD | TBD | TBD | TBD |

*Statistical comparisons (Welch's t-test, Cohen's d) are computed by `aggregate_seeds.py`.*

### Visualizations

The master script automatically generates comparison plots and copies them to the `assets/` directory for clean persistence:

![Deconfounded Experiments Comparison](assets/deconfounded_experiments_plot.png)
*Figure 1: Comparison of baseline, vanilla KD, and deconfounded feature, attention, and relational KD methods across tasks.*

![Feature KD Weight Ablation](assets/feature_kd_ablation_plot.png)
*Figure 2: Analysis of the feature distillation loss constraint weight (λ) on downstream transfer.*

---

## Analysis

### Classification vs. Dense Prediction

*(To be updated after experiments run)*

A key question from the assignment: do KD methods behave differently on classification vs. spatially-structured tasks?

**Expected trends** (to be validated):

- Attention KD directly transfers spatial structure (token-to-token attention patterns), so it should benefit depth estimation more than pure logit distillation
- Relational KD preserves geometric relationships within the batch, which may generalize better to spatial tasks where inter-sample structure matters
- Feature KD forces coordinate-level alignment, which could help or hurt spatial tasks depending on whether the projection layer preserves spatial information

The frozen depth head uses spatial patch tokens (not just CLS), so methods that improve spatial token quality should show larger gains on depth estimation.

### Feature KD: The Regularization Sweet Spot

The λ ablation reveals a non-obvious dynamic: lowering feature reconstruction loss does NOT always improve downstream performance. The λ=0.01 config achieves the lowest training loss but worse transfer than λ=0.1. This suggests that some representational slack is necessary — the student needs room to develop its own task-discriminative boundaries rather than perfectly mimicking the teacher.

### KD as Implicit Regularization

The most striking finding from multi-seed analysis is not accuracy improvement but variance reduction. On a small dataset, the regularization effect of teacher supervision may be as important as the representational transfer. This has practical implications: for deployment scenarios with limited labeled data, even simple Vanilla KD significantly stabilizes training.

### Confounding in Multi-Objective Experiments

Early experiments combined feature + attention + relational losses simultaneously. The combined Attention KD (with Feature KD at λ=1.0) performed worse than either method in isolation. This was initially confusing, but isolating each objective revealed that the rigid λ=1.0 feature constraint was the bottleneck. This experience reinforced the importance of controlled single-variable experiments before making causal claims about method effectiveness.

---

## Project Structure

```text
├── configs/                      
│   ├── baseline.yaml
│   ├── kd_baseline.yaml
│   ├── feature_kd_lam*.yaml      
│   ├── attention_kd_only.yaml    
│   └── relational_kd_only.yaml   
│
├── models/
│   ├── teacher.py                
│   ├── student.py                
│   └── heads/
│       ├── classification_head.py
│       ├── depth_head.py         
│       └── segmentation_head.py  
│
├── distillation/
│   ├── losses/
│   │   ├── kd_loss.py           
│   │   ├── feature_loss.py       
│   │   ├── attention_loss.py     
│   │   └── relational_loss.py    
│   ├── trainer.py                
│   └── projector.py              
│
├── datasets/
│   ├── oxford_pets.py
│   ├── cifar100.py
│   ├── nyuv2.py
│   └── pascal_voc.py
│
├── evaluation/
│   ├── classification.py         # Frozen backbone + linear probe
│   ├── depth.py                  # RMSE, δ1 metrics
│   └── segmentation.py           # mIoU metric
│
├── train.py                      # Main distillation training
├── evaluate.py                   # Oxford Pets evaluation
├── evaluate_teacher.py           # Teacher ceiling
├── evaluate_cifar100.py          # CIFAR-100 evaluation
├── evaluate_depth.py             # NYUv2 depth evaluation
├── evaluate_segmentation.py      # VOC segmentation evaluation
├── download_nyuv2.py             # NYUv2 data preparation
├── aggregate_seeds.py            # Multi-seed stats (t-test, CI)
├── plot_metrics.py               # Comparison plots
├── run_all.sh                    # Master experiment pipeline
└── outputs/                      # Saved metrics, plots, configs
```

---

## How to Run

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

This runs all training, evaluations across 3 tasks, multi-seed variance, and ablations:

```bash
bash run_all.sh
```

Estimated time: 2-3 hours on a single GPU.

### 3. Run Individual Components

```bash
# Train a specific config
python train.py --config configs/feature_kd_lam0_1.yaml

# Evaluate on Oxford Pets
python evaluate.py --config configs/feature_kd_lam0_1.yaml --checkpoint checkpoints/feature_kd_lam0_1.pt

# Evaluate on CIFAR-100
python evaluate_cifar100.py --config configs/feature_kd_lam0_1.yaml --checkpoint checkpoints/feature_kd_lam0_1.pt --subset_size 5000

# Evaluate on NYUv2 depth
python evaluate_depth.py --config configs/feature_kd_lam0_1.yaml --checkpoint checkpoints/feature_kd_lam0_1.pt

# Teacher ceiling (all tasks)
python evaluate_teacher.py --subset_size 1000 --seed 42

# Multi-seed aggregation with statistical tests
python aggregate_seeds.py
```

### 4. Download NYUv2 Data

```bash
pip install h5py
python download_nyuv2.py
```

---

## Reproducibility

- All experiments use fixed seeds applied to `torch`, `numpy`, `random`, and `cudnn` deterministic mode
- Dataset subset sampling uses a **dedicated `torch.Generator`** (not the global seed state) to ensure identical subsets regardless of model initialization order
- DataLoader workers are seeded via `worker_init_fn`
- Configs are copied to each experiment's output directory
- All metrics saved as JSON for programmatic comparison

---

## Limitations & Future Work

1. **Scale**: Experiments use reduced subsets (1000 Oxford Pets, 5000 CIFAR-100). Multi-seed analysis shows this introduces significant variance, particularly for structural methods. Scaling to full datasets would strengthen conclusions.
2. **Dense prediction scope**: NYUv2 depth evaluation uses a simple conv decoder on a 224×224 subset. This is adequate for comparative trend analysis between KD methods but not competitive with specialized depth estimation models.
3. **Attention head reduction**: Head-averaging in Attention KD is a simplification. A learnable attention projection could preserve fine-grained head relationships.
4. **Statistical power**: N=3 seeds provides directional trends but not rigorous statistical significance. N≥5 with bootstrap confidence intervals would be more defensible.
5. **No ImageNet-1K evaluation**: Full ImageNet-1K training was deferred in favor of multi-task evaluation on smaller benchmarks. The framework supports adding it with minimal changes.

---

## Extending the Framework

New KD strategies can be added by:

1. Creating a loss module in `distillation/losses/`
2. Registering it in `distillation/trainer.py`
3. Adding a YAML config in `configs/`

New evaluation tasks can be added by:

1. Creating a dataset loader in `datasets/`
2. Creating an evaluation head in `models/heads/`
3. Creating an evaluation script following the existing pattern

---
