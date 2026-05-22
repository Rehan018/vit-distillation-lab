import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_experiment_metrics(output_dirs, save_path='outputs/comparison_plot.png'):
    plt.figure(figsize=(12, 5))
    

    plt.subplot(1, 2, 1)
    for exp_dir in output_dirs:
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        train_metrics_path = os.path.join(exp_dir, 'train_metrics.json')
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path, 'r') as f:
                data = json.load(f)
            
                epochs = data.get('epochs_loss', [data.get('final_train_loss')])
                plt.plot(epochs, marker='o', label=exp_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch / Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    

    plt.subplot(1, 2, 2)
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
        plt.bar(exp_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orchid', 'orange'][:len(exp_names)])
        for i, val in enumerate(accuracies):
            plt.text(i, val + 0.5, f"{val:.2f}%", ha='center', fontweight='bold')
    plt.title('Oxford Pets Linear Probe Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot successfully saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot distillation metrics")
    parser.add_argument('--dirs', nargs='+', required=True, help="List of experiment output directories to compare")
    parser.add_argument('--save_path', type=str, default='outputs/comparison_plot.png', help="Output plot image path")
    args = parser.parse_args()
    plot_experiment_metrics(args.dirs, args.save_path)
