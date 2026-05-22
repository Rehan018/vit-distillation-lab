
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
from datasets.nyuv2 import get_nyuv2_dataloaders
from evaluation.depth import evaluate_depth

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
    parser = argparse.ArgumentParser(description="NYUv2 Depth Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--nyuv2_root", type=str, default="./data/nyuv2")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "teacher":
        model = TeacherModel(model_name=config['teacher']['model'])
        embed_dim = config['teacher']['embed_dim']
    else:
        model = StudentModel(model_name=config['student']['model'])
        embed_dim = config['student']['embed_dim']
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt['student_state_dict'])

    train_loader, test_loader = get_nyuv2_dataloaders(
        root_dir=args.nyuv2_root, batch_size=16, img_size=224,
        subset_size=args.subset_size, seed=seed,
    )

    metrics = evaluate_depth(model=model, train_loader=train_loader, test_loader=test_loader,
                             device=device, embed_dim=embed_dim, epochs=args.epochs)

    
    exp_name = config['experiment']['name']
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    results_path = f"{out_dir}/eval_metrics.json"
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing = json.load(f)
    existing['depth_rmse'] = round(metrics['rmse'], 4)
    existing['depth_delta1'] = round(metrics['delta1'], 2)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=4)
    logger.info(f"Depth metrics saved to {results_path}")


if __name__ == "__main__":
    main()
