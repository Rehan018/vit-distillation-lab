
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import numpy as np
from models.heads.segmentation_head import SegmentationHead

logger = logging.getLogger(__name__)
IGNORE_INDEX = 255


def compute_miou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        if not target_cls.any():
            continue
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return (sum(ious) / max(1, len(ious))) * 100.0


def evaluate_segmentation(model, train_loader, val_loader, device, embed_dim, num_classes=21, epochs=10, lr=1e-3):
    model.eval()
    model.to(device)

    head = SegmentationHead(embed_dim=embed_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.Adam(head.parameters(), lr=lr)

    logger.info(f"Training Segmentation Head for {epochs} epochs...")
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Seg Epoch {epoch+1}/{epochs}")
        for rgb, mask in pbar:
            rgb, mask = rgb.to(device), mask.to(device)
            with torch.no_grad():
                features = model(rgb, return_features=True)
            optimizer.zero_grad()
            logits = head(features)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        logger.info(f"Seg Epoch {epoch+1}/{epochs} — Loss: {total_loss / max(1, n_batches):.4f}")

    logger.info("Evaluating Segmentation on validation set...")
    head.eval()
    all_miou = []
    with torch.no_grad():
        for rgb, mask in tqdm(val_loader, desc="Seg Eval"):
            rgb, mask = rgb.to(device), mask.to(device)
            features = model(rgb, return_features=True)
            logits = head(features)
            preds = logits.argmax(dim=1)
            valid = mask != IGNORE_INDEX
            preds_valid = preds.clone()
            preds_valid[~valid] = IGNORE_INDEX
            miou = compute_miou(preds_valid, mask, num_classes)
            all_miou.append(miou)

    avg_miou = sum(all_miou) / max(1, len(all_miou))
    logger.info(f"Segmentation Results — mIoU: {avg_miou:.2f}%")
    return {'miou': avg_miou}
