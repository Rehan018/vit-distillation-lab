
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from typing import Dict
from models.heads.depth_head import DepthPredictionHead

logger = logging.getLogger(__name__)


def compute_depth_metrics(pred, target, min_depth=1e-3):
    valid_mask = target > min_depth
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    if len(pred_valid) == 0:
        return {'rmse': 0.0, 'delta1': 0.0}
    rmse = torch.sqrt(((pred_valid - target_valid) ** 2).mean()).item()
    ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
    delta1 = (ratio < 1.25).float().mean().item() * 100.0
    return {'rmse': rmse, 'delta1': delta1}


def evaluate_depth(model, train_loader, test_loader, device, embed_dim, epochs=10, lr=1e-3):

    model.eval()
    model.to(device)

    head = DepthPredictionHead(embed_dim=embed_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(head.parameters(), lr=lr)

    logger.info(f"Training Depth Head for {epochs} epochs...")
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Depth Epoch {epoch+1}/{epochs}")
        for rgb, depth_gt in pbar:
            rgb, depth_gt = rgb.to(device), depth_gt.to(device)
            with torch.no_grad():
                features = model(rgb, return_features=True)
            optimizer.zero_grad()
            depth_pred = head(features)
            loss = criterion(depth_pred, depth_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        logger.info(f"Depth Epoch {epoch+1}/{epochs} — Loss: {total_loss / max(1, n_batches):.4f}")

    logger.info("Evaluating Depth Head on test set...")
    head.eval()
    all_rmse, all_delta1 = [], []
    with torch.no_grad():
        for rgb, depth_gt in tqdm(test_loader, desc="Depth Eval"):
            rgb, depth_gt = rgb.to(device), depth_gt.to(device)
            features = model(rgb, return_features=True)
            depth_pred = head(features)
            metrics = compute_depth_metrics(depth_pred, depth_gt)
            all_rmse.append(metrics['rmse'])
            all_delta1.append(metrics['delta1'])

    avg_rmse = sum(all_rmse) / max(1, len(all_rmse))
    avg_delta1 = sum(all_delta1) / max(1, len(all_delta1))
    logger.info(f"Depth Results — RMSE: {avg_rmse:.4f}, δ1: {avg_delta1:.2f}%")
    return {'rmse': avg_rmse, 'delta1': avg_delta1}
