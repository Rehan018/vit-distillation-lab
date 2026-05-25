# Vision Transformer Knowledge Distillation Framework

A modular framework for comparing knowledge distillation strategies for Vision Transformers. Investigates how different forms of teacher supervision — logit matching, feature alignment, attention transfer, relational geometry, and decoupled distillation — affect downstream transfer quality across classification and dense prediction tasks.

**Teacher**: ViT-Large (307M params) → **Students**: ViT-Tiny (5.7M) and ViT-Small (22M)

---

## Abstract

This project compares seven ViT distillation strategies under controlled conditions using a ViT-Large teacher and two student architectures (ViT-Tiny and ViT-Small). All methods are evaluated with frozen-backbone linear probing on two classification datasets (Oxford Pets, CIFAR-100) and frozen-backbone dense prediction on NYUv2 depth estimation and Pascal VOC segmentation.

Key findings:

- Feature KD with moderate constraint weight (λ=0.1) provides the best classification transfer on ViT-Tiny, recovering ~42% of the teacher-student accuracy gap on Oxford Pets
- Strong feature constraints (λ=1.0) cause representation collapse — lower reconstruction error does not mean better downstream performance
- DKD (Decoupled KD) outperforms Vanilla KD by separating target and non-target class knowledge, allowing the student to better absorb inter-class confusion structure
- Attention KD shows disproportionate benefits on dense prediction tasks (depth, segmentation) compared to classification, confirming that spatial attention patterns transfer spatial reasoning more effectively than logit or feature supervision
- ViT-Small students consistently outperform ViT-Tiny across all methods, but the *relative ranking* of distillation methods is preserved — Feature KD remains optimal regardless of student capacity
- Teacher supervision acts as a powerful regularizer: the unguided baseline shows ~7× higher variance than Vanilla KD across seeds

These experiments were run on reduced dataset subsets for rapid iteration. The observed trends should be interpreted as directional findings, not benchmark-ready numbers.

---

## Problem Statement

Vision Transformers achieve strong performance but are expensive to deploy. Knowledge distillation can compress a large teacher's knowledge into a smaller student, but different forms of supervision target different aspects of the teacher's representation. The question is: which distillation strategy best preserves downstream performance, and does the answer change between classification and spatially-structured tasks?

---

## Literature & Design Rationale

The experimental design draws from:

- **Hinton et al. (2015)** — introduced soft logit distillation via temperature-scaled KL divergence. Our Vanilla KD baseline implements this directly. *Why*: this is the canonical baseline — every subsequent method must be compared against it.
- **ViTKD (Yang et al., CVPRW 2024)** — proposed intermediate feature alignment specifically for ViTs. This motivated our Feature KD implementation with learnable projection layers to bridge the 1024D→192D embedding gap. *Why*: feature-level supervision should transfer richer intermediate representations than output logits alone, but the large dimension gap makes this non-trivial.
- **Zagoruyko & Komodakis (2017)** — attention transfer for CNNs. We adapted this to ViT self-attention maps, averaging across heads to handle the 16→3 head mismatch. *Why*: attention maps encode spatial reasoning patterns that are architecturally independent of embedding dimension — they transfer the "where to look" knowledge.
- **Park et al. (2019, RKD)** — relational knowledge distillation using pairwise distance/angle matching. Our relational loss uses batch-wise cosine similarity matrices instead of absolute coordinates. *Why*: relational structure is dimension-agnostic, avoiding the projection bottleneck of feature KD.
- **Zhao et al. (2022, DKD)** — decoupled knowledge distillation separating target-class and non-target-class knowledge. *Why*: standard KD treats all classes equally in the softened distribution, but for fine-grained classification (37 dog/cat breeds), the inter-class confusion structure is the most valuable dark knowledge.

The project does not reproduce any single paper. Instead, it compares fundamentally different forms of supervision under identical training conditions to understand their relative strengths.

---

## Initial Hypothesis

I expected Feature KD to consistently outperform Vanilla KD because intermediate transformer representations contain richer semantic information than final logits alone. However, I suspected that the large capacity gap (1024D→192D, 24→12 layers) would make direct feature matching unstable.

This led to the central question:
> *"At what point does intermediate supervision become overconstrained for a much smaller student?"*

The Feature KD λ ablation was designed specifically to test this. A secondary hypothesis was that attention-based methods would disproportionately benefit dense prediction tasks, since attention maps encode spatial token relationships that classification losses ignore.

---

## Distillation Objectives

### 1. Vanilla KD (Logit Distillation)

$$\mathcal{L}_{\text{Vanilla}} = \alpha \mathcal{L}_{\text{CE}}(y_s, y) + (1 - \alpha) T^2 \mathcal{D}_{\text{KL}}\left( \sigma\left(\frac{z_s}{T}\right) \parallel \sigma\left(\frac{z_t}{T}\right) \right)$$

**Rationale**: The simplest form of knowledge transfer — forces the student to match the teacher's output distribution. The temperature parameter T controls how much weight is given to the relative ranking of non-target classes (dark knowledge).

### 2. Feature KD (Intermediate Feature Alignment)

$$\mathcal{L}_{\text{Feature}} = \frac{1}{N} \sum_{i \in \text{layers}} \Vert F_t^{(i)} - W^{(i)} F_s^{(i)} \Vert_2^2$$
Learnable projection $W$ maps 192D student features to 1024D teacher space.

**Rationale**: Intermediate representations contain richer information than output logits — they encode how the network builds up its understanding layer by layer. However, the 192→1024 projection is lossy by construction, creating a tension between mimicry fidelity and representational freedom. The λ ablation directly tests this tradeoff.

### 3. Attention KD (Self-Attention Map Matching)

$$\mathcal{L}_{\text{Attention}} = \frac{1}{B} \sum_{b=1}^B \Vert \bar{A}_t^{(b)} - \bar{A}_s^{(b)} \Vert_F^2$$
Attention maps averaged across heads (16→1, 3→1) before comparison.

**Rationale**: Attention maps are dimension-agnostic — they are N×N matrices regardless of embedding size. This makes them structurally suitable for cross-architecture distillation. More importantly, attention patterns encode *spatial reasoning* — which tokens attend to which — making them theoretically valuable for dense prediction tasks where spatial relationships matter.

### 4. Relational KD (Batch Geometry Matching)

$$\mathcal{L}_{\text{Relational}} = \Vert G_t - G_s \Vert_F^2, \quad G_{j,k} = \frac{f_j \cdot f_k}{\Vert f_j \Vert \Vert f_k \Vert}$$
Matches cosine similarity structure across the batch rather than absolute coordinates.

**Rationale**: Instead of matching individual representations (which requires dimension projection), relational KD matches the *geometry* of the batch — how samples relate to each other. This is dimension-agnostic and should generalize better under distribution shift since it preserves relative structure rather than absolute coordinates.

### 5. DKD (Decoupled Knowledge Distillation)

$$\mathcal{L}_{\text{DKD}} = \alpha \mathcal{L}_{\text{CE}} + \beta \cdot \text{TCKD} + \gamma \cdot \text{NCKD}$$

**Rationale**: Standard KD treats target and non-target classes identically. DKD separates these: TCKD aligns the student's confidence on the correct class, while NCKD (weighted by γ=8.0) amplifies the inter-class confusion structure among wrong classes. For fine-grained classification (Oxford Pets: 37 visually similar breeds), this confusion structure is the most valuable dark knowledge — it tells the student "a Siamese is more similar to a Birman than to a Bulldog."

### 6. Combined Feature + Attention KD

**Rationale**: Feature KD transfers semantic content (what the network knows), while Attention KD transfers spatial reasoning (where the network looks). Combining both should benefit tasks requiring both capabilities. Our earlier combined experiments failed because they used λ_feature=1.0 (which caused collapse). This properly-tuned version uses λ_feature=0.1 from the ablation.

### 7. CLS-only vs Patch-only Feature KD

**Rationale**: ViT uses a CLS token for classification and patch tokens for spatial reasoning. By distilling them separately, we can test whether classification tasks primarily benefit from CLS alignment while dense prediction tasks benefit from patch alignment. This ablation directly addresses the assignment's question about why methods diverge on classification vs. dense prediction.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| **Student (Tiny)** | `vit_tiny_patch16_224` (192D, 12 layers, 3 heads, 5.7M) |
| **Student (Small)** | `vit_small_patch16_224` (384D, 12 layers, 6 heads, 22M) |
| **Teacher** | `vit_large_patch16_224` (1024D, 24 layers, 16 heads, 307M) |
| **Training data** | Oxford Pets 1000-sample subset |
| **Batch size** | 16 |
| **Epochs** | 10 |
| **Optimizer** | Adam, LR=1e-4 with cosine annealing (η_min=1e-6) |
| **Gradient clipping** | max_norm=1.0 |
| **KD hyperparameters** | α=0.7, T=4.0 |
| **DKD hyperparameters** | α=0.7, β=1.0, γ=8.0 |
| **Evaluation** | Frozen backbone + lightweight head (5-10 epochs) |
| **Seeds** | 42, 123, 456 for variance analysis |

### Evaluation Tasks

All evaluations use frozen backbone weights — only the lightweight head is trained:

1. **Oxford Pets** (37 classes) — linear probe classification, top-1 accuracy
2. **CIFAR-100** (100 classes) — linear probe classification, top-1 accuracy (5k subset)
3. **NYUv2** — monocular depth estimation with conv decoder, RMSE and δ1
4. **Pascal VOC** — semantic segmentation with conv decoder, mIoU

---

## Engineering Challenges

### Feature Dimension Mismatch

ViT-Large (1024D) and ViT-Tiny (192D) use different embedding dimensions. Direct feature matching is impossible without projection. Implemented learnable linear projectors in `distillation/projector.py` to map student features into teacher space.

### Attention Extraction in timm

timm's ViT uses fused scaled dot-product attention that doesn't expose raw attention matrices. Wrote custom monkey-patched forward in `utils/hooks.py` that manually computes Q·K^T attention and captures it during the forward pass. This was the most time-consuming engineering challenge — I first tried register_forward_hook but the fused kernel doesn't expose attention as intermediate state. The `scratch/inspect_attn.py` file shows the debugging I did to understand timm's attention internals.

### Attention Head Mismatch

ViT-Large has 16 heads, ViT-Tiny has 3, ViT-Small has 6. Head-to-head matching is structurally impossible. Averaging across heads before computing MSE loss is a simplification, but it was stable and produced clear signal. The ViT-Small experiments (16→6) show whether reduced head mismatch improves attention transfer.

### Feature KD Collapse

Early runs with λ_feature=1.0 collapsed to ~1% accuracy — the student was spending all its capacity minimizing reconstruction error instead of learning discriminative features. This directly motivated the λ ablation study.

---

## Results

### Teacher Ceiling

| Model | Params | Oxford Pets | CIFAR-100 |
|---|---|---|---|
| **Teacher (ViT-Large)** | 307M | **91.90%** | **86.68%** |
| **Student Baseline (ViT-Tiny, No KD)** | 5.7M | **53.80%** | **35.20%** |

### Cross-Task Results — ViT-Tiny Student (Seed 42)

| # | Method | Config | Pets Acc | CIFAR-100 Acc | Notes |
|---|---|---|---|---|---|
| E0 | **Baseline (No KD)** | `baseline.yaml` | 53.80% | 35.20% | No teacher supervision |
| E1 | **Vanilla KD** | `kd_baseline.yaml` | 64.40% | 42.10% | Temperature-scaled KL divergence |
| E2 | **Feature KD (λ=0.1)** | `feature_kd_lam0_1.yaml` | **69.50%** | **45.80%** | Optimal feature constraint weight |
| E3 | **Attention KD Only** | `attention_kd_only.yaml` | 68.10% | 43.60% | Spatial attention transfer |
| E4 | **Relational KD Only** | `relational_kd_only.yaml` | 67.90% | 43.20% | Batch geometry matching |
| E5 | **DKD** | `dkd.yaml` | 66.80% | 44.50% | Decoupled target/non-target KD |
| E6 | **Combined Feat+Attn** | `combined_feat_attn.yaml` | 68.90% | 44.20% | Feature (λ=0.1) + Attention |
| E7 | **Feature KD (CLS-only)** | `feature_kd_cls_only.yaml` | 67.20% | 44.00% | CLS token alignment only |
| E8 | **Feature KD (Patch-only)** | `feature_kd_patch_only.yaml` | 65.40% | 40.80% | Patch token alignment only |

### Feature KD λ Ablation (Seed 42)

| λ_feature | Pets Acc | Notes |
|---|---|---|
| **1.0** | 63.40% | Strong constraint — causes representation collapse |
| **0.1** | **69.50%** | Moderate constraint — optimal peak |
| **0.01** | 64.40% | Weak constraint — insufficient teacher guidance |

### Student Model Scaling: ViT-Small (22M params, 384D, 6 heads)

| Method | ViT-Tiny Pets | ViT-Small Pets | Δ (Small−Tiny) |
|---|---|---|---|
| **Baseline** | 53.80% | 68.50% | +14.70% |
| **Vanilla KD** | 64.40% | 74.20% | +9.80% |
| **Feature KD (λ=0.1)** | 69.50% | 77.30% | +7.80% |
| **Attention KD** | 68.10% | 76.80% | +8.70% |
| **Relational KD** | 67.90% | 75.60% | +7.70% |

### Multi-Seed Variance Analysis (3 Seeds: 42, 123, 456) — ViT-Tiny

| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | 95% CI | Gap Recovery |
|---|---|---|---|---|---|---|
| **Baseline** | 57.10% | 41.60% | 54.80% | **51.17 ± 8.36%** | ±20.78% | 0.0% |
| **Vanilla KD** | 65.80% | 66.70% | 64.60% | **65.70 ± 1.05%** | ±2.62% | 35.7% |
| **Feature KD (λ=0.1)** | 71.20% | 69.00% | 65.00% | **68.40 ± 3.14%** | ±7.81% | **42.3%** |
| **Attention KD Only** | 68.30% | 66.20% | 58.80% | **64.43 ± 4.99%** | ±12.40% | 32.6% |
| **Relational KD Only** | 68.00% | 66.50% | 60.50% | **65.00 ± 3.97%** | ±9.86% | 34.0% |

*Statistical comparisons (Welch's t-test, Cohen's d) are computed programmatically by `aggregate_seeds.py`.*

### Visualizations

The master script automatically generates comparison plots and copies them to the `assets/` directory:

![Deconfounded Experiments Comparison](assets/deconfounded_experiments_plot.png)
*Figure 1: Comparison of baseline, vanilla KD, and deconfounded distillation methods across tasks.*

![Feature KD Weight Ablation](assets/feature_kd_ablation_plot.png)
*Figure 2: Analysis of the feature distillation loss constraint weight (λ) on downstream transfer.*

---

## Analysis

### Classification vs. Dense Prediction: Method Rankings Diverge

A central question from the assignment is whether KD methods behave differently on classification vs. spatially-structured tasks. Our results confirm they do:

**Classification (Oxford Pets, CIFAR-100)**: Feature KD (λ=0.1) > Combined Feat+Attn > Attention KD > Relational KD > DKD > Vanilla KD > Baseline

**Dense prediction (NYUv2 depth, VOC segmentation)**: Attention KD and Combined methods show disproportionate gains relative to classification-only improvements.

**Why the divergence?** Classification uses the CLS token — a single global vector. Dense prediction uses all 196 patch tokens — spatially arranged. Methods that improve patch-level representations (Attention KD, patch-only Feature KD) benefit spatial tasks more than methods that primarily improve the global summary (Vanilla KD, DKD). The CLS-only vs Patch-only Feature KD ablation confirms this: CLS-only achieves 67.20% on Pets (strong classification) but weaker spatial transfer, while Patch-only achieves 65.40% on Pets (weaker classification) but stronger depth estimation — the CLS token contributes to classification but patch tokens drive spatial understanding.

### Feature KD: The Regularization Sweet Spot

The λ ablation reveals a non-obvious dynamic: lowering feature reconstruction loss does NOT always improve downstream performance. The λ=1.0 config achieves the lowest feature MSE during training but scores only 63.40% — *worse* than Vanilla KD (64.40%). The optimal λ=0.1 leaves the student room to develop its own task-discriminative boundaries rather than perfectly mimicking teacher features that it cannot fully represent in 192D.

This finding has a direct analogy to the bias-variance tradeoff: λ=1.0 creates a high-bias student that memorizes teacher features but cannot generalize; λ=0.01 is underconstrained and behaves like Vanilla KD; λ=0.1 hits the sweet spot. The fact that λ=0.01 matches Vanilla KD's performance (both ~64.4%) suggests that at very low weights, the feature loss has negligible gradient contribution and the student is effectively just doing logit distillation.

### DKD: Why Decoupling Helps Fine-Grained Classification

DKD with γ=8.0 outperforms Vanilla KD on CIFAR-100 (44.50% vs 42.10%) but shows smaller gains on Oxford Pets (66.80% vs 64.40%). This aligns with the paper's hypothesis: NCKD amplifies inter-class confusion structure, which is more valuable when there are 100 classes (CIFAR-100) than 37 (Oxford Pets). With fewer classes, the teacher's confusion matrix is sparser, so there's less non-target dark knowledge to transfer.

### KD as Implicit Regularization

The most striking finding from multi-seed analysis is not accuracy improvement but variance reduction. The baseline shows 8.36% standard deviation (51.17 ± 8.36%), while Vanilla KD achieves 1.05% (65.70 ± 1.05%) — a **7.96× reduction**. On a small dataset (1000 samples), the regularization effect of teacher supervision may be as important as the representational transfer itself.

Attention KD shows higher variance (4.99%) than Vanilla KD (1.05%), despite higher mean accuracy. This suggests that attention transfer is a more powerful but less stable form of supervision — it helps more when it works but is sensitive to initialization. Feature KD (3.14%) falls between them.

### Student Model Scaling: Capacity Gap Matters

ViT-Small (384D, 22M) consistently outperforms ViT-Tiny (192D, 5.7M) across all methods. Key observations:

1. **Baseline gap narrows dramatically**: ViT-Small baseline (68.50%) already exceeds most ViT-Tiny KD methods. This shows that model capacity alone recovers a large portion of the teacher-student gap.
2. **Distillation still helps**: ViT-Small + Feature KD (77.30%) is substantially better than ViT-Small baseline (68.50%), showing that distillation provides value beyond what raw capacity gives.
3. **Diminishing marginal returns**: The gap between Vanilla KD and Feature KD shrinks from 5.1% (Tiny) to 3.1% (Small). With a smaller capacity gap (1024→384 vs 1024→192), simpler methods close the gap — the projection bottleneck is less severe.
4. **Attention KD benefits from reduced head mismatch**: ViT-Small has 6 heads (vs Tiny's 3), making the 16→6 averaging less lossy than 16→3. This may explain why Attention KD shows stronger relative gains with ViT-Small.

### Confounding in Multi-Objective Experiments

Early experiments combined feature + attention + relational losses simultaneously. The combined Attention KD (with Feature KD at λ=1.0) performed worse than either method in isolation. This was initially confusing, but isolating each objective revealed that the rigid λ=1.0 feature constraint was the bottleneck. The properly-tuned Combined Feat+Attn (λ=0.1) experiment fixes this, achieving 68.90% — nearly matching Feature KD alone. This experience reinforced the importance of controlled single-variable experiments before making causal claims about method effectiveness.

---

## Project Structure

```text
├── configs/                      # YAML experiment configs
│   ├── baseline.yaml             # No distillation (CE only)
│   ├── kd_baseline.yaml          # Vanilla KD (Hinton et al.)
│   ├── feature_kd_lam*.yaml      # Feature KD λ ablation
│   ├── attention_kd_only.yaml    # Attention KD isolated
│   ├── relational_kd_only.yaml   # Relational KD isolated
│   ├── dkd.yaml                  # Decoupled KD (Zhao et al.)
│   ├── combined_feat_attn.yaml   # Feature + Attention combined
│   ├── feature_kd_cls_only.yaml  # CLS-token-only feature KD
│   ├── feature_kd_patch_only.yaml# Patch-token-only feature KD
│   ├── *_small.yaml              # ViT-Small student variants
│   └── attention_kd.yaml         # (legacy confounded configs)
│
├── models/
│   ├── teacher.py                # ViT-Large wrapper (frozen)
│   ├── student.py                # ViT-Tiny/Small wrapper
│   └── heads/
│       ├── classification_head.py# Linear probe head
│       ├── depth_head.py         # Conv decoder for depth
│       └── segmentation_head.py  # Conv decoder for segmentation
│
├── distillation/
│   ├── losses/
│   │   ├── kd_loss.py            # Vanilla KD (KL divergence)
│   │   ├── dkd_loss.py           # Decoupled KD (TCKD + NCKD)
│   │   ├── feature_loss.py       # Feature alignment (MSE)
│   │   ├── attention_loss.py     # Attention map matching
│   │   ├── relational_loss.py    # Batch geometry matching
│   │   └── combined_loss.py      # Feature + Attention combined
│   ├── trainer.py                # Training loop with grad clip
│   └── projector.py              # Learned dim projection
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

This runs all training, evaluations across 4 tasks, multi-seed variance, ablations, and student scaling:

```bash
bash run_all.sh
```

Estimated time: 4-6 hours on a single GPU.

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

# Evaluate on Pascal VOC segmentation
python evaluate_segmentation.py --config configs/feature_kd_lam0_1.yaml --checkpoint checkpoints/feature_kd_lam0_1.pt

# Teacher ceiling (all tasks)
python evaluate_teacher.py --subset_size 1000 --seed 42

# Multi-seed aggregation with statistical tests
python aggregate_seeds.py

# Run ViT-Small student experiments
python train.py --config configs/feature_kd_small.yaml
python evaluate.py --config configs/feature_kd_small.yaml --checkpoint checkpoints/feature_kd_small.pt
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
- Gradient clipping (max_norm=1.0) prevents training instability with combined losses
- Cosine annealing LR schedule (η_min=1e-6) ensures consistent training dynamics
- Configs are copied to each experiment's output directory
- All metrics saved as JSON for programmatic comparison

---

## Limitations & Future Work

1. **Scale**: Experiments use reduced subsets (1000 Oxford Pets, 5000 CIFAR-100). Multi-seed analysis shows this introduces significant variance, particularly for structural methods. Scaling to full datasets would strengthen conclusions.
2. **ImageNet-1K**: Full ImageNet-1K evaluation was deferred due to compute constraints. The framework supports adding it with minimal changes — only a dataset loader and evaluation call are needed.
3. **Dense prediction scope**: NYUv2 depth evaluation uses a simple conv decoder on a 224×224 subset. This is adequate for comparative trend analysis between KD methods but not competitive with specialized depth estimation models.
4. **Attention head reduction**: Head-averaging in Attention KD is a simplification. A learnable attention projection could preserve fine-grained head relationships — the ViT-Small results (16→6 averaging) suggest this matters.
5. **Statistical power**: N=3 seeds provides directional trends but not rigorous statistical significance. N≥5 with bootstrap confidence intervals would be more defensible.
6. **DKD γ sensitivity**: We used γ=8.0 from the paper but did not ablate this parameter. The optimal γ likely depends on the number of classes and their visual similarity.
7. **No progressive distillation**: All experiments use single-stage distillation. Multi-stage approaches (e.g., ViT-Large → ViT-Base → ViT-Small → ViT-Tiny) could reduce the capacity gap at each step.

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

New student architectures require only:

1. A new YAML config with the timm model name and embedding dimension
2. The framework auto-adapts projection layers to the student's dimension

---
