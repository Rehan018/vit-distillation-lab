#!/bin/bash
set -e

# =============================================================================
# Master Experiment Pipeline
#
# Re-runs critical experiments with the fixed evaluation protocol:
#   - Deterministic subset sampling (dedicated Generator)
#   - CLS token linear probing (standard ViT protocol)
#   - Gradient clipping + cosine LR scheduling
#
# Then evaluates all methods on:
#   1. Oxford Pets classification (linear probe)
#   2. CIFAR-100 classification (linear probe)
#   3. NYUv2 depth estimation (frozen backbone + depth head)
#
# Run time estimate: ~2-3 hours on single GPU
# =============================================================================

echo "================================================================"
echo "  Full Experiment Pipeline (Fixed Protocol)"
echo "================================================================"

# ----- STEP 0: Data Preparation -----
echo ""
echo "=== Step 0: Preparing Datasets ==="
echo "  Oxford Pets: auto-downloaded by torchvision"
echo "  CIFAR-100: auto-downloaded by torchvision"
echo ""

# NYUv2 requires manual download (2.8GB .mat file)
if [ ! -d "./data/nyuv2/train/rgb" ]; then
    echo "  NYUv2: Downloading and extracting (this takes a while)..."
    pip install h5py -q
    python download_nyuv2.py
else
    echo "  NYUv2: Already prepared"
fi

# ----- STEP 1: Teacher Ceiling -----
echo ""
echo "=== Step 1: Teacher Ceiling (ViT-Large) ==="
python evaluate_teacher.py --subset_size 1000 --seed 42

# Evaluate teacher on CIFAR-100 and NYUv2 too
echo "  Evaluating teacher on CIFAR-100..."
python evaluate_cifar100.py \
    --config configs/baseline.yaml \
    --model_type teacher \
    --subset_size 5000

echo "  Evaluating teacher on NYUv2 depth..."
python evaluate_depth.py \
    --config configs/baseline.yaml \
    --model_type teacher \
    --epochs 10

# ----- STEP 2: Train Core Experiments -----
# Only re-train the 5 key configs (not all ablations)
CORE_EXPERIMENTS=(
    "baseline:configs/baseline.yaml"
    "vanilla_kd:configs/kd_baseline.yaml"
    "feature_kd_lam0_1:configs/feature_kd_lam0_1.yaml"
    "attention_kd_only:configs/attention_kd_only.yaml"
    "relational_kd_only:configs/relational_kd_only.yaml"
)

for entry in "${CORE_EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name config_path <<< "$entry"

    echo ""
    echo "=== Training: ${exp_name} ==="
    python train.py --config "$config_path"

    echo "  Evaluating on Oxford Pets..."
    python evaluate.py --config "$config_path" --checkpoint "checkpoints/${exp_name}.pt"

    echo "  Evaluating on CIFAR-100..."
    python evaluate_cifar100.py \
        --config "$config_path" \
        --checkpoint "checkpoints/${exp_name}.pt" \
        --subset_size 5000

    echo "  Evaluating on NYUv2 depth..."
    python evaluate_depth.py \
        --config "$config_path" \
        --checkpoint "checkpoints/${exp_name}.pt" \
        --epochs 10

    echo "  Completed: ${exp_name}"
done

# ----- STEP 3: Multi-Seed Variance (3 critical methods only) -----
echo ""
echo "=== Step 3: Multi-Seed Variance Analysis ==="

SEED_EXPERIMENTS=(
    "baseline:configs/baseline.yaml"
    "vanilla_kd:configs/kd_baseline.yaml"
    "feature_kd_lam0_1:configs/feature_kd_lam0_1.yaml"
)

SEEDS=(42 123 456)

for entry in "${SEED_EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name config_path <<< "$entry"

    for seed in "${SEEDS[@]}"; do
        run_name="${exp_name}_seed${seed}"
        echo ""
        echo "--- ${run_name} ---"

        # Create temp config with seed override
        tmp_config="/tmp/${run_name}.yaml"
        cp "$config_path" "$tmp_config"
        sed -i "s/^  name: .*/  name: ${run_name}/" "$tmp_config"
        sed -i "s/^  seed: .*/  seed: ${seed}/" "$tmp_config"

        python train.py --config "$tmp_config"
        python evaluate.py --config "$tmp_config" --checkpoint "checkpoints/${run_name}.pt"

        rm -f "$tmp_config"
    done
done

# Aggregate seed results
python aggregate_seeds.py

# ----- STEP 4: Feature KD Lambda Ablation -----
echo ""
echo "=== Step 4: Feature KD λ Ablation ==="

for lam in "feature_kd_lam1_0" "feature_kd_lam0_01"; do
    echo "  Training ${lam}..."
    python train.py --config "configs/${lam}.yaml"
    python evaluate.py --config "configs/${lam}.yaml" --checkpoint "checkpoints/${lam}.pt"
done

# ----- STEP 5: Generate Plots -----
echo ""
echo "=== Step 5: Generating Plots ==="

python plot_metrics.py \
    --dirs outputs/teacher_ceiling outputs/baseline outputs/vanilla_kd \
           outputs/feature_kd_lam0_1 outputs/attention_kd_only outputs/relational_kd_only \
    --save_path outputs/deconfounded_experiments_plot.png

python plot_metrics.py \
    --dirs outputs/feature_kd_lam1_0 outputs/feature_kd_lam0_1 outputs/feature_kd_lam0_01 \
    --save_path outputs/feature_kd_ablation_plot.png

# Copy plots to assets/ so they are tracked by git and can be shown in README.md
cp outputs/deconfounded_experiments_plot.png assets/
cp outputs/feature_kd_ablation_plot.png assets/
echo "  Plots copied to assets/ for README visualization"

echo ""
echo "================================================================"
echo "  All experiments complete!"
echo "  Results saved to outputs/ and assets/"
echo "  Run 'python aggregate_seeds.py' to see variance summary"
echo "================================================================"

