
import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from utils.config import load_config
from models.student import StudentModel
from models.teacher import TeacherModel
from datasets.cifar100 import get_cifar100_dataloaders
from evaluation.classification import evaluate_linear_probe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Classification Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to student checkpoint (omit for teacher)")
    parser.add_argument("--model_type", type=str, default="student",
                        choices=["student", "teacher"],
                        help="Evaluate student or teacher model")
    parser.add_argument("--subset_size", type=int, default=5000,
                        help="Subset size (default 5000 — CIFAR-100 has 50k train)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs for linear probe training")
    return parser.parse_args()


def run():
    args = main()
    config = load_config(args.config)

    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

   
    if args.model_type == "teacher":
        logger.info("Loading Teacher Model for CIFAR-100 evaluation...")
        model = TeacherModel(model_name=config['teacher']['model'])
        embed_dim = config['teacher']['embed_dim']
    else:
        logger.info("Loading Student Model for CIFAR-100 evaluation...")
        model = StudentModel(model_name=config['student']['model'])
        embed_dim = config['student']['embed_dim']

        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}...")
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt['student_state_dict'])


    logger.info(f"Loading CIFAR-100 (subset={args.subset_size})...")
    train_loader, test_loader, num_classes = get_cifar100_dataloaders(
        root_dir='./data',
        batch_size=64,
        img_size=224,
        subset_size=args.subset_size,
        seed=seed,
    )

    
    logger.info("Starting CIFAR-100 Linear Probe Evaluation...")
    accuracy = evaluate_linear_probe(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        num_classes=num_classes,
        embed_dim=embed_dim,
        epochs=args.epochs,
        lr=1e-3,
        pooling="cls",
    )

    logger.info(f"CIFAR-100 Accuracy: {accuracy:.2f}%")

    exp_name = config['experiment']['name']
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    results_path = f"{out_dir}/eval_metrics.json"
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing = json.load(f)

    existing['cifar100_acc'] = round(accuracy, 2)

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=4)

    logger.info(f"CIFAR-100 metrics saved to {results_path}")


if __name__ == "__main__":
    run()
