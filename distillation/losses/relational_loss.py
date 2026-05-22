import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalDistillationLoss(nn.Module):
    def __init__(self, normalize=True, pooled=True):
        super().__init__()
        self.normalize = normalize
        self.pooled = pooled

    def forward(self, student_features, teacher_features):
        
        loss = 0.0
        num_layers = len(student_features)
        
        for layer_name in student_features.keys():
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]
            
            if self.pooled:

                s_feat = s_feat.mean(dim=1)
                t_feat = t_feat.mean(dim=1)
            else:
                s_feat = s_feat.view(s_feat.size(0), -1)
                t_feat = t_feat.view(t_feat.size(0), -1)
                
            if self.normalize:
                s_feat = F.normalize(s_feat, p=2, dim=-1)
                t_feat = F.normalize(t_feat, p=2, dim=-1)
                

            s_sim = s_feat @ s_feat.transpose(-1, -2)
            t_sim = t_feat @ t_feat.transpose(-1, -2)
            
            loss += F.mse_loss(s_sim, t_sim)
            
        return loss / max(1, num_layers)
