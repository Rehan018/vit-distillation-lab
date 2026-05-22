import torch
import torch.nn as nn

class FeatureProjector(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)

class MultiLayerProjector(nn.Module):
    def __init__(self, student_dim, teacher_dim, layer_names):
        super().__init__()
    
        self.projectors = nn.ModuleDict({
            name: FeatureProjector(student_dim, teacher_dim)
            for name in layer_names
        })
        
    def forward(self, student_features):
      
        projected_features = {}
        for name, feat in student_features.items():
            projected_features[name] = self.projectors[name](feat)
        return projected_features
