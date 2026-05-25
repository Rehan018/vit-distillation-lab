import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalDistillationLoss(nn.Module):
    """Relational Knowledge Distillation loss.
    
    Implements distance-wise and angle-wise relational distillation from
    Park et al., "Relational Knowledge Distillation", CVPR 2019.
    
    Instead of matching absolute representations (which is prone to capacity matching issues),
    RKD matches the geometric relations (distance and triplet angle structures) among data
    points in the batch.
    
    Why: Relational structure is highly invariant to the absolute capacity mismatch of the student
    (192D vs 1024D) and helps generalization under distribution shifts because it preserves 
    manifold topology rather than coordinate locations.
    """
    def __init__(self, normalize=True, pooled=True, w_dist=1.0, w_angle=2.0):
        super().__init__()
        self.normalize = normalize  # Preserved for config compatibility
        self.pooled = pooled
        self.w_dist = w_dist
        self.w_angle = w_angle

    def _pairwise_distances(self, x):
        # Efficient pairwise Euclidean distance matrix computation
        sq_norms = torch.sum(x ** 2, dim=-1, keepdim=True)
        dist_sq = sq_norms + sq_norms.transpose(-1, -2) - 2 * (x @ x.transpose(-1, -2))
        dist_sq = torch.clamp(dist_sq, min=1e-12)
        return torch.sqrt(dist_sq)

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
                
            # Compute pairwise distances
            s_dist = self._pairwise_distances(s_feat)
            t_dist = self._pairwise_distances(t_feat)
            
            # RKD distance-wise loss normalization: divide by average pairwise distance
            s_mu = s_dist.mean() + 1e-8
            t_mu = t_dist.mean() + 1e-8
            s_dist_norm = s_dist / s_mu
            t_dist_norm = t_dist / t_mu
            
            # Smooth L1 is recommended by paper for distance regression
            dist_loss = F.smooth_l1_loss(s_dist_norm, t_dist_norm, reduction='mean')
            
            # RKD angle-wise loss: B x B x D difference vectors
            # s_diff[i, j] represents vector from sample j to sample i
            s_diff = s_feat.unsqueeze(1) - s_feat.unsqueeze(0)
            t_diff = t_feat.unsqueeze(1) - t_feat.unsqueeze(0)
            
            # Normalize diff vectors
            s_diff_norm = F.normalize(s_diff, p=2, dim=-1)
            t_diff_norm = F.normalize(t_diff, p=2, dim=-1)
            
            # Vectorized dot product of normalized diff vectors over all triplets (i, j, k)
            # yielding cos angle between (i, j) and (k, j) at anchor j.
            # Shape is B(i) x B(j) x B(k)
            s_cos = torch.einsum('ijd,kjd->ijk', s_diff_norm, s_diff_norm)
            t_cos = torch.einsum('ijd,kjd->ijk', t_diff_norm, t_diff_norm)
            
            # Smooth L1 for angle regression
            angle_loss = F.smooth_l1_loss(s_cos, t_cos, reduction='mean')
            
            loss += self.w_dist * dist_loss + self.w_angle * angle_loss
            
        return loss / max(1, num_layers)

