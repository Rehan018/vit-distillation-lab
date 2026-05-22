import argparse
import random
import numpy as np
import torch
from utils.config import load_config
from models.student import StudentModel
from datasets.oxford_pets import get_oxford_pets_dataloaders
from evaluation.classification import evaluate_linear_probe
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Distillation Models")
    parser.add_argument("--config", type=str, default="configs/relational_kd.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading Student Model...")
    student = StudentModel(model_name=config['student']['model'])
    
    
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    student.load_state_dict(checkpoint['student_state_dict'])
    
    logger.info("Loading Oxford-IIIT Pets Dataset...")
    train_loader, val_loader, num_classes = get_oxford_pets_dataloaders(
        root_dir='./data',
        batch_size=64,
        img_size=224,
        subset_size=config['training'].get('subset_size', None)
    )
    
    logger.info("Starting Linear Probe Evaluation...")
    accuracy = evaluate_linear_probe(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        embed_dim=config['student']['embed_dim'],
        epochs=5,
        lr=1e-3
    )
    
    logger.info(f"Evaluation Complete! Final Oxford Pets Accuracy: {accuracy:.2f}%")
    
    import json
    import os
    exp_name = config['experiment']['name']
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_file = f"{out_dir}/eval_metrics.json"
    metrics = {"val_acc": accuracy}
    if os.path.exists(f"{out_dir}/train_metrics.json"):
        with open(f"{out_dir}/train_metrics.json", "r") as f:
            metrics.update(json.load(f))
            
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
