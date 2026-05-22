import torch
import torch.nn as nn
from tqdm import tqdm

class DistillationTrainer:
    def __init__(self, teacher, student, criterion, optimizer, device,
                 feature_criterion=None, projector=None,
                 teacher_extractor=None, student_extractor=None,
                 attention_criterion=None, teacher_attn_extractor=None, student_attn_extractor=None,
                 relational_criterion=None,
                 lambda_kd=1.0, lambda_feat=1.0, lambda_attn=1.0, lambda_rel=1.0):
        self.teacher = teacher
        self.student = student
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.feature_criterion = feature_criterion
        
      
        self.projector = projector
        if self.projector is not None:
            self.projector.to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters()})
            
        self.teacher_extractor = teacher_extractor
        self.student_extractor = student_extractor
        
        self.attention_criterion = attention_criterion
        self.teacher_attn_extractor = teacher_attn_extractor
        self.student_attn_extractor = student_attn_extractor
        
        self.relational_criterion = relational_criterion
        
        self.lambda_kd = lambda_kd
        self.lambda_feat = lambda_feat
        self.lambda_attn = lambda_attn
        self.lambda_rel = lambda_rel
        
        self.teacher.to(self.device)
        self.teacher.eval() 
        
        self.student.to(self.device)

    def train_epoch(self, dataloader):
        self.student.train()
        if self.projector:
            self.projector.train()
            
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                inputs = batch.to(self.device)
                targets = None
            
            self.optimizer.zero_grad()
            
 
            if self.teacher_extractor:
                self.teacher_extractor.clear()
            if self.student_extractor:
                self.student_extractor.clear()
                
            if self.teacher_attn_extractor:
                self.teacher_attn_extractor.clear()
            if self.student_attn_extractor:
                self.student_attn_extractor.clear()

            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
                
            student_outputs = self.student(inputs)
    
            loss = self.criterion(student_outputs, teacher_outputs, targets) * self.lambda_kd
            
            feat_loss = None
            rel_loss = None
            attn_loss = None
    
            if (self.feature_criterion or self.relational_criterion) and self.teacher_extractor and self.student_extractor and self.projector:
                t_feat = self.teacher_extractor.features
                s_feat = self.student_extractor.features
           
                s_feat_proj = self.projector(s_feat)
                

                if self.feature_criterion:
                    feat_loss = self.feature_criterion(s_feat_proj, t_feat)
                    loss += self.lambda_feat * feat_loss
   
                if self.relational_criterion:
                    rel_loss = self.relational_criterion(s_feat_proj, t_feat)
                    loss += self.lambda_rel * rel_loss

            if self.attention_criterion and self.teacher_attn_extractor and self.student_attn_extractor:
                t_attn = self.teacher_attn_extractor.attention_maps
                s_attn = self.student_attn_extractor.attention_maps
                attn_loss = self.attention_criterion(s_attn, t_attn)
                loss += self.lambda_attn * attn_loss
            
 
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            postfix = {"loss": f"{loss.item():.4f}"}
            if feat_loss is not None:
                postfix["feat_loss"] = f"{feat_loss.item():.4f}"
            if attn_loss is not None:
                postfix["attn_loss"] = f"{attn_loss.item():.4f}"
            if rel_loss is not None:
                postfix["rel_loss"] = f"{rel_loss.item():.4f}"
            pbar.set_postfix(postfix)
            
        return total_loss / len(dataloader)

    def save_checkpoint(self, path, config=None, metrics=None):
    
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'config': config,
            'metrics': metrics
        }, path)
