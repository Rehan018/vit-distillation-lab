
import os
import json
import numpy as np
from scipy import stats


EXPERIMENTS = [
    "baseline",
    "vanilla_kd",
    "feature_kd_lam0_1",
    "attention_kd_only",
    "relational_kd_only",
]

SEEDS = [42, 123, 456]

DISPLAY_NAMES = {
    "baseline": "Baseline (No KD)",
    "vanilla_kd": "Vanilla KD",
    "feature_kd_lam0_1": "Feature KD (λ=0.1)",
    "attention_kd_only": "Attention KD Only",
    "relational_kd_only": "Relational KD Only",
}


def cohens_d(group1: list, group2: list) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('nan')
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    results = {}
    baseline_accs = []

    for exp in EXPERIMENTS:
        accuracies = []
        for seed in SEEDS:
            run_name = f"{exp}_seed{seed}"
            eval_path = f"outputs/{run_name}/eval_metrics.json"

            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    data = json.load(f)
                    acc = data.get("val_acc", None)
                    if acc is not None:
                        accuracies.append(acc)
                        print(f"  {run_name}: {acc:.2f}%")
            else:
                print(f"  WARNING: Missing {eval_path}")

        if accuracies:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0

            n = len(accuracies)
            if n > 1:
                t_crit = stats.t.ppf(0.975, df=n-1)
                ci_half = t_crit * (std_acc / np.sqrt(n))
            else:
                ci_half = 0.0

            results[exp] = {
                "mean": round(mean_acc, 2),
                "std": round(std_acc, 2),
                "ci_95": round(ci_half, 2),
                "runs": accuracies,
                "n_seeds": len(accuracies),
            }

            if exp == "baseline":
                baseline_accs = accuracies

            print(f"  → {DISPLAY_NAMES[exp]}: {mean_acc:.2f} ± {std_acc:.2f}% "
                  f"(95% CI: ±{ci_half:.2f}%)")
        else:
            print(f"  → {DISPLAY_NAMES[exp]}: NO DATA")

    if baseline_accs and len(baseline_accs) >= 2:
        print("\n## Statistical Comparisons vs Baseline\n")
        print("| Method | Mean Diff | Cohen's d | t-stat | p-value | Significant? |")
        print("|---|---|---|---|---|---|")

        for exp in EXPERIMENTS:
            if exp == "baseline" or exp not in results:
                continue

            method_accs = results[exp]["runs"]
            if len(method_accs) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(method_accs, baseline_accs, equal_var=False)
            d = cohens_d(method_accs, baseline_accs)
            mean_diff = np.mean(method_accs) - np.mean(baseline_accs)
            sig = "Yes*" if p_value < 0.05 else "No"

            results[exp]["vs_baseline"] = {
                "mean_diff": round(float(mean_diff), 2),
                "cohens_d": round(float(d), 2),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_value), 4),
                "significant_p05": bool(p_value < 0.05),
            }

            print(f"| {DISPLAY_NAMES[exp]} | +{mean_diff:.2f}% | {d:.2f} | "
                  f"{t_stat:.3f} | {p_value:.4f} | {sig} |")

        print("\n*Note: With N=3 seeds, statistical power is limited. "
              "Results should be interpreted as directional trends rather than "
              "definitive rankings. p < 0.05 with N=3 indicates a strong effect, "
              "but wider seed sweeps (N≥5) would be needed for rigorous conclusions.*")

    out_path = "outputs/multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAggregated results saved to {out_path}")

    teacher_acc = 91.9 
    print("\n## Multi-Seed Results Summary\n")
    print("| Method | Seed 42 | Seed 123 | Seed 456 | Mean ± Std | 95% CI | Gap Recovery |")
    print("|---|---|---|---|---|---|---|")

    for exp in EXPERIMENTS:
        if exp in results:
            r = results[exp]
            runs = r["runs"]
            seed_vals = [f"{v:.1f}%" for v in runs] + ["—"] * (3 - len(runs))
            gap_recovery = ((r["mean"] - results.get("baseline", {}).get("mean", 53.8))
                            / (teacher_acc - results.get("baseline", {}).get("mean", 53.8))) * 100
            print(f"| {DISPLAY_NAMES[exp]} | {seed_vals[0]} | {seed_vals[1]} | "
                  f"{seed_vals[2]} | **{r['mean']:.2f} ± {r['std']:.2f}%** | "
                  f"±{r['ci_95']:.2f}% | {gap_recovery:.1f}% |")


if __name__ == "__main__":
    main()
