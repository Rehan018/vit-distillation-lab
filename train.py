import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.config import load_config
from models.teacher import TeacherModel
from models.student import StudentModel
from distillation.trainer import DistillationTrainer
from distillation.losses.kd_loss import KDLoss
from distillation.losses.feature_loss import FeatureDistillationLoss
from distillation.losses.attention_loss import AttentionDistillationLoss
from distillation.losses.relational_loss import RelationalDistillationLoss
from distillation.projector import MultiLayerProjector
from utils.hooks import FeatureHookManager, AttentionHookManager
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
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--config", type=str, default="configs/relational_kd.yaml", help="Path to the config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading Teacher...")
    teacher = TeacherModel(model_name=config['teacher']['model'])
    
    logger.info("Loading Student...")
    student = StudentModel(model_name=config['student']['model'])
    
    # Hooks and Projectors
    teacher_extractor = None
    student_extractor = None
    feature_criterion = None
    projector = None
    
    teacher_attn_extractor = None
    student_attn_extractor = None
    attention_criterion = None
    
    layer_names = []
    
    # Feature KD Setup
    if config.get('feature_kd', {}).get('enabled', False) or config.get('attention_kd', {}).get('enabled', False):
        t_layers = config['feature_kd']['teacher_layers']
        s_layers = config['feature_kd']['student_layers']
        
        for t_idx, s_idx in zip(t_layers, s_layers):
            layer_names.append(f"mapped_{t_idx}_{s_idx}")
            
    if config.get('feature_kd', {}).get('enabled', False):
        logger.info("Setting up Feature Hooks...")
        teacher_extractor = FeatureHookManager()
        student_extractor = FeatureHookManager()
        
        for idx, (t_idx, s_idx) in enumerate(zip(t_layers, s_layers)):
            layer_name = layer_names[idx]
            teacher_extractor.register_hook(teacher.model.blocks[t_idx], layer_name)
            student_extractor.register_hook(student.model.blocks[s_idx], layer_name)
            
        logger.info("Setting up Feature Projector...")
        projector = MultiLayerProjector(
            student_dim=config['student']['embed_dim'],
            teacher_dim=config['teacher']['embed_dim'],
            layer_names=layer_names
        )
        
        feature_criterion = FeatureDistillationLoss(
            distill_cls=config['feature_kd']['distill_cls'],
            distill_patch=config['feature_kd']['distill_patch']
        )
        
    if config.get('attention_kd', {}).get('enabled', False):
        logger.info("Setting up Attention Hooks...")
        teacher_attn_extractor = AttentionHookManager()
        student_attn_extractor = AttentionHookManager()
        
        for idx, (t_idx, s_idx) in enumerate(zip(t_layers, s_layers)):
            layer_name = layer_names[idx]
            teacher_attn_extractor.patch_attention(teacher.model.blocks[t_idx].attn, layer_name)
            student_attn_extractor.patch_attention(student.model.blocks[s_idx].attn, layer_name)
            
        attention_criterion = AttentionDistillationLoss(
            average_heads=config['attention_kd']['average_heads']
        )
        
    relational_criterion = None
    if config.get('relational_kd', {}).get('enabled', False):
        logger.info("Setting up Relational Distillation...")
        relational_criterion = RelationalDistillationLoss(
            normalize=config['relational_kd']['normalize'],
            pooled=config['relational_kd']['pooled']
        )
    
    logger.info("Verifying shapes with dummy data...")
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)
        
    logger.info(f"Teacher logit shape: {teacher_out.shape}")
    logger.info(f"Student logit shape: {student_out.shape}")
    
    if teacher_extractor and student_extractor:
        logger.info("Verifying intermediate feature shapes...")
        for name in teacher_extractor.features.keys():
            t_f = teacher_extractor.features[name]
            s_f = student_extractor.features[name]
            logger.info(f"Layer {name} - Teacher feature shape: {t_f.shape}, Student feature shape: {s_f.shape}")

    if teacher_attn_extractor and student_attn_extractor:
        logger.info("Verifying attention map shapes...")
        for name in teacher_attn_extractor.attention_maps.keys():
            t_a = teacher_attn_extractor.attention_maps[name]
            s_a = student_attn_extractor.attention_maps[name]
            logger.info(f"Layer {name} - Teacher attention shape: {t_a.shape}, Student attention shape: {s_a.shape}")

    import os
    import json
    import shutil
    
    exp_name = config['experiment']['name']
    out_dir = f"outputs/{exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    shutil.copy(args.config, f"{out_dir}/config.yaml")

    from datasets.oxford_pets import get_oxford_pets_dataloaders
    logger.info("Loading Oxford-IIIT Pets Dataset for Distillation...")
    train_loader, val_loader, _ = get_oxford_pets_dataloaders(
        root_dir='./data',
        batch_size=config['training'].get('batch_size', 32),
        img_size=224,
        subset_size=config['training'].get('subset_size', None)
    )
    
    # Setup criterion and optimizer
    criterion = KDLoss(temperature=config['distillation']['temperature'], alpha=config['distillation']['alpha'])
    optimizer = optim.Adam(student.parameters(), lr=config['training']['lr'])
    
    # Setup trainer
    trainer = DistillationTrainer(
        teacher=teacher, 
        student=student, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        feature_criterion=feature_criterion,
        projector=projector,
        teacher_extractor=teacher_extractor,
        student_extractor=student_extractor,
        attention_criterion=attention_criterion,
        teacher_attn_extractor=teacher_attn_extractor,
        student_attn_extractor=student_attn_extractor,
        relational_criterion=relational_criterion,
        lambda_kd=config.get('feature_kd', {}).get('lambda_kd', 1.0),
        lambda_feat=config.get('feature_kd', {}).get('lambda_feature', 0.0),
        lambda_attn=config.get('attention_kd', {}).get('lambda_attn', 0.0),
        lambda_rel=config.get('relational_kd', {}).get('lambda_rel', 0.0)
    )
    
    logger.info("Starting distillation training loop...")
    epochs = config['training']['epochs']
    loss_trajectory = []
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_loader)
        loss_trajectory.append(loss)
        logger.info(f"Epoch {epoch+1}/{epochs} - Total Loss: {loss:.4f}")
        
    metrics = {
        'final_train_loss': loss_trajectory[-1] if loss_trajectory else 0.0,
        'epochs_loss': loss_trajectory
    }
    with open(f"{out_dir}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Checkpoint
    checkpoint_path = f"checkpoints/{exp_name}.pt"
    trainer.save_checkpoint(
        path=checkpoint_path, 
        config=config, 
        metrics=metrics
    )
    logger.info(f"Saved student checkpoint to {checkpoint_path}")
        
    logger.info(f"Training complete for experiment: {exp_name}")

if __name__ == "__main__":
    main()
