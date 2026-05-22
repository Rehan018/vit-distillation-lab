
import argparse
import random
import numpy as np
import torch
from models.teacher import TeacherModel
from datasets.oxford_pets import get_oxford_pets_dataloaders
from evaluation.classification import evaluate_linear_probe
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Evaluate Teacher Model (Ceiling)")
    parser.add_argument("--subset_size", type=int, default=1000, help="Subset size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading Teacher Model (ViT-Large)...")
    teacher = TeacherModel(model_name="vit_large_patch16_224")

    logger.info("Loading Oxford-IIIT Pets Dataset...")
    train_loader, val_loader, num_classes = get_oxford_pets_dataloaders(
        root_dir='./data',
        batch_size=64,
        img_size=224,
        subset_size=args.subset_size
    )

    logger.info("Starting Teacher Linear Probe Evaluation...")
    accuracy = evaluate_linear_probe(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        embed_dim=1024, 
        epochs=5,
        lr=1e-3
    )

    logger.info(f"Teacher Ceiling Accuracy: {accuracy:.2f}%")

    
    out_dir = "outputs/teacher_ceiling"
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"val_acc": accuracy, "model": "vit_large_patch16_224", "subset_size": args.subset_size}
    with open(f"{out_dir}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Teacher ceiling metrics saved to {out_dir}/eval_metrics.json")

if __name__ == "__main__":
    main()
