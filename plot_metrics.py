import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_experiment_metrics(output_dirs, save_path='outputs/comparison_plot.png'):
    # Determine the number of panels we need.
    # If any experiment contains sub-losses, we will add a third panel showing sub-loss curves.
    has_sub_losses = False
    sub_loss_data = {}
    
    for exp_dir in output_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        train_metrics_path = os.path.join(exp_dir, 'train_metrics.json')
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path, 'r') as f:
                data = json.load(f)
                
            # Check if any trajectory has non-zero elements
            feat_traj = data.get('epochs_feat_loss', [])
            attn_traj = data.get('epochs_attn_loss', [])
            rel_traj = data.get('epochs_rel_loss', [])
            
            if (any(v > 0 for v in feat_traj) or 
                any(v > 0 for v in attn_traj) or 
                any(v > 0 for v in rel_traj)):
                has_sub_losses = True
                sub_loss_data[exp_name] = {
                    'feat': feat_traj,
                    'attn': attn_traj,
                    'rel': rel_traj,
                    'kd': data.get('epochs_kd_loss', [])
                }

    n_cols = 3 if has_sub_losses else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    
    if n_cols == 1:
        axes = [axes]
        
    # --- PANEL 1: Training Loss ---
    ax_loss = axes[0]
    for exp_dir in output_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        train_metrics_path = os.path.join(exp_dir, 'train_metrics.json')
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path, 'r') as f:
                data = json.load(f)
            
                epochs = data.get('epochs_loss', [data.get('final_train_loss')])
                ax_loss.plot(epochs, marker='o', linewidth=2, label=exp_name)
    ax_loss.set_title('Global Training Loss Alignment', fontsize=12, fontweight='bold', pad=10)
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.legend(frameon=True, facecolor='white', edgecolor='none')
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    
    # --- PANEL 2: Downstream Accuracy ---
    ax_acc = axes[1]
    exp_names = []
    accuracies = []
    for exp_dir in output_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        eval_metrics_path = os.path.join(exp_dir, 'eval_metrics.json')
        if os.path.exists(eval_metrics_path):
            with open(eval_metrics_path, 'r') as f:
                data = json.load(f)
                acc = data.get('val_acc', 0)
                exp_names.append(exp_name)
                accuracies.append(acc)
                
    if exp_names:
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f']
        bars = ax_acc.bar(exp_names, accuracies, color=colors[:len(exp_names)], edgecolor='none', alpha=0.85, width=0.6)
        for bar in bars:
            height = bar.get_height()
            ax_acc.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
                        
    ax_acc.set_title('Oxford Pets Probe Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.6)
    plt.setp(ax_acc.get_xticklabels(), rotation=30, ha='right')

    # --- PANEL 3: Distillation Loss Component Alignment (Feature/Attention/Relational curves) ---
    if has_sub_losses:
        ax_sub = axes[2]
        # Plot alignment curves for the first experiment that contains them
        for exp_name, curves in sub_loss_data.items():
            if any(v > 0 for v in curves['feat']):
                ax_sub.plot(curves['feat'], marker='s', linestyle='-', linewidth=2, label=f'{exp_name} (Feature Loss)')
            if any(v > 0 for v in curves['attn']):
                ax_sub.plot(curves['attn'], marker='^', linestyle='--', linewidth=2, label=f'{exp_name} (Attention Loss)')
            if any(v > 0 for v in curves['rel']):
                ax_sub.plot(curves['rel'], marker='d', linestyle='-.', linewidth=2, label=f'{exp_name} (Relational Loss)')
            if any(v > 0 for v in curves['kd']):
                ax_sub.plot(curves['kd'], marker='o', linestyle=':', linewidth=1.5, alpha=0.7, label=f'{exp_name} (KD/Task Loss)')
                
        ax_sub.set_title('Feature & Objective Alignment Curves', fontsize=12, fontweight='bold', pad=10)
        ax_sub.set_xlabel('Epoch', fontsize=10)
        ax_sub.set_ylabel('Sub-loss Value', fontsize=10)
        ax_sub.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=8)
        ax_sub.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # Create parent folder if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot successfully saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot distillation metrics")
    parser.add_argument('--dirs', nargs='+', required=True, help="List of experiment output directories to compare")
    parser.add_argument('--save_path', type=str, default='outputs/comparison_plot.png', help="Output plot image path")
    args = parser.parse_args()
    plot_experiment_metrics(args.dirs, args.save_path)

