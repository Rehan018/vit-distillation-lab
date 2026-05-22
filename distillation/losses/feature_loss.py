import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    def __init__(self, distill_cls=True, distill_patch=True):
        super().__init__()
        self.distill_cls = distill_cls
        self.distill_patch = distill_patch

    def forward(self, student_features, teacher_features):
        loss = 0.0
        num_layers = len(student_features)
        
        for layer_name in student_features.keys():
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]
            
            layer_loss = 0.0
            
            if self.distill_cls and self.distill_patch:
                layer_loss = F.mse_loss(s_feat, t_feat)
            elif self.distill_cls:
                layer_loss = F.mse_loss(s_feat[:, 0], t_feat[:, 0])
            elif self.distill_patch:
                layer_loss = F.mse_loss(s_feat[:, 1:], t_feat[:, 1:])
                
            loss += layer_loss
            
        return loss / max(1, num_layers)
