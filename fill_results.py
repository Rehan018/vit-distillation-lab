"""
Run all CIFAR-100 evaluations and auto-fill README.md results.

Evaluates: Teacher, Baseline, Vanilla KD, Feature KD lam0_1,
           Attention KD Only, Relational KD Only, Feature KD lam1_0, lam0_01

Then reads all outputs/*.json and patches the README tables.
"""
import subprocess
import json
import os
import sys
import re

VENV_PYTHON = ".venv/bin/python"
SUBSET = 5000
EPOCHS = 5

EXPERIMENTS = [
    ("teacher",           "configs/baseline.yaml",          "teacher",  None),
    ("baseline",          "configs/baseline.yaml",          "student",  "checkpoints/baseline.pt"),
    ("vanilla_kd",        "configs/kd_baseline.yaml",       "student",  "checkpoints/vanilla_kd.pt"),
    ("feature_kd_lam0_1", "configs/feature_kd_lam0_1.yaml", "student",  "checkpoints/feature_kd_lam0_1.pt"),
    ("attention_kd_only", "configs/attention_kd_only.yaml", "student",  "checkpoints/attention_kd_only.pt"),
    ("relational_kd_only","configs/relational_kd_only.yaml","student",  "checkpoints/relational_kd_only.pt"),
    ("feature_kd_lam1_0", "configs/feature_kd_lam1_0.yaml", "student",  "checkpoints/feature_kd_lam1_0.pt"),
    ("feature_kd_lam0_01","configs/feature_kd_lam0_01.yaml","student",  "checkpoints/feature_kd_lam0_01.pt"),
]

def run_cifar100(exp_name, config, model_type, checkpoint):
    out_file = f"outputs/{exp_name}/eval_metrics.json"
    if os.path.exists(out_file):
        with open(out_file) as f:
            data = json.load(f)
        if "cifar100_acc" in data:
            print(f"  [SKIP] {exp_name}: cifar100_acc={data['cifar100_acc']}% (already done)")
            return data["cifar100_acc"]

    cmd = [
        VENV_PYTHON, "evaluate_cifar100.py",
        "--config", config,
        "--model_type", model_type,
        "--subset_size", str(SUBSET),
        "--epochs", str(EPOCHS),
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]

    print(f"\n{'='*60}")
    print(f"  Running CIFAR-100: {exp_name} ({model_type})")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {exp_name} failed!")
        return None

    with open(out_file) as f:
        data = json.load(f)
    acc = data.get("cifar100_acc")
    print(f"  -> {exp_name} CIFAR-100: {acc}%")
    return acc


def patch_readme(results):
    """Patch README.md tables with actual values."""
    with open("README.md", "r") as f:
        content = f.read()

    teacher_pets = "91.90%"
    baseline_pets = "53.80%"
    teacher_c100 = f"{results.get('teacher', 'TBD')}%" if results.get('teacher') else "TBD"
    baseline_c100 = f"{results.get('baseline', 'TBD')}%" if results.get('baseline') else "TBD"

    content = re.sub(
        r'\| \*\*Teacher \(ViT-Large\)\*\* \| 307M \| \*\*91\.90%\*\* \| TBD \| TBD \| TBD \|',
        f'| **Teacher (ViT-Large)** | 307M | **91.90%** | **{teacher_c100}** | TBD | TBD |',
        content
    )
    content = re.sub(
        r'\| \*\*Student Baseline \(No KD\)\*\* \| 5\.7M \| \*\*53\.80%\*\* \| TBD \| TBD \| TBD \|',
        f'| **Student Baseline (No KD)** | 5.7M | **53.80%** | **{baseline_c100}** | TBD | TBD |',
        content
    )
    def patch_row(text, method_label, config_label, pets_acc, c100_acc):
        escaped_method = re.escape(method_label)
        escaped_config = re.escape(config_label)
        c100_str = f"**{c100_acc}%**" if c100_acc else "TBD"
        pattern = rf'(\| {escaped_method} \| {escaped_config} \| [^|]+ \|) TBD (\| TBD \| TBD \|)'
        replacement = rf'\g<1> {c100_str} \2'
        return re.sub(pattern, replacement, text)

    rows = [
        ("**Baseline**",          "`baseline.yaml`",          results.get("baseline")),
        ("**Vanilla KD**",        "`kd_baseline.yaml`",       results.get("vanilla_kd")),
        ("**Feature KD (λ=0.1)**","`feature_kd_lam0_1.yaml`", results.get("feature_kd_lam0_1")),
        ("**Attention KD Only**", "`attention_kd_only.yaml`", results.get("attention_kd_only")),
        ("**Relational KD Only**","`relational_kd_only.yaml`",results.get("relational_kd_only")),
    ]
    for method, config_label, acc in rows:
        if acc:
            content = patch_row(content, method, config_label, None, acc)

    with open("README.md", "w") as f:
        f.write(content)

    print("\n✓ README.md patched with actual CIFAR-100 results!")


def main():
    results = {}
    for exp_name, config, model_type, checkpoint in EXPERIMENTS:
        if model_type == "student" and checkpoint and not os.path.exists(checkpoint):
            print(f"  [SKIP] {exp_name}: checkpoint {checkpoint} not found")
            continue
        acc = run_cifar100(exp_name, config, model_type, checkpoint)
        if acc is not None:
            results[exp_name] = acc

    print("\n" + "="*60)
    print("  CIFAR-100 Results Summary")
    print("="*60)
    for k, v in results.items():
        print(f"  {k:30s}: {v:.2f}%")

    patch_readme(results)
    return results


if __name__ == "__main__":
    main()
