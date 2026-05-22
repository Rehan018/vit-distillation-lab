import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from models.heads.classification_head import LinearClassificationHead

logger = logging.getLogger(__name__)


def _pool_features(features: torch.Tensor, strategy: str = "cls") -> torch.Tensor:
   
    if len(features.shape) == 2:
        return features
    if strategy == "cls":
        return features[:, 0]  
    elif strategy == "mean":
        return features.mean(dim=1)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def evaluate_linear_probe(model, train_loader, val_loader, device, num_classes, embed_dim,
                          epochs=10, lr=1e-3, pooling="cls"):
    model.eval()
    model.to(device)
    
    head = LinearClassificationHead(in_features=embed_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=lr)
    
    logger.info(f"Training Linear Probe for {epochs} epochs (pooling={pooling})...")
    
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = model(images, return_features=True)
                features = _pool_features(features, strategy=pooling)
            
            optimizer.zero_grad()
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
            
    logger.info("Validating Linear Probe...")
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating Probe"):
            images, labels = images.to(device), labels.to(device)
            features = model(images, return_features=True)
            features = _pool_features(features, strategy=pooling)
            outputs = head(features)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    logger.info(f"Final Validation Accuracy: {accuracy:.2f}%")
    return accuracy
