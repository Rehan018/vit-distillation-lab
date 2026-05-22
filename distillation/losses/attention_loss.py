import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDistillationLoss(nn.Module):
    def __init__(self, average_heads=True):
        super().__init__()
        self.average_heads = average_heads

    def forward(self, student_attention, teacher_attention):
        loss = 0.0
        num_layers = len(student_attention)
        
        for layer_name in student_attention.keys():
            s_attn = student_attention[layer_name]
            t_attn = teacher_attention[layer_name]
            
            if self.average_heads:
            
                s_attn = s_attn.mean(dim=1)
                t_attn = t_attn.mean(dim=1)
                
            loss += F.mse_loss(s_attn, t_attn)
            
        return loss / max(1, num_layers)
