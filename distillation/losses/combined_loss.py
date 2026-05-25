"""
Combined Feature + Attention Distillation Loss (Proper Implementation)

Rationale: Our initial combined experiments (attention_kd.yaml, relational_kd.yaml) were
confounded because they used λ_feature=1.0, which caused representation collapse. This
module provides a properly weighted combination of feature alignment and attention transfer,
using the optimal λ_feature=0.1 discovered in the ablation study.

Why combine both: Feature KD forces the student's intermediate representations toward the
teacher's embedding space (good for semantic content), while Attention KD transfers spatial
attention patterns (good for localization). The hypothesis is that combining both at proper
weights should outperform either alone, especially on dense prediction tasks where both
spatial structure and semantic content matter.

This is conceptually similar to the multi-objective approach in ViTKD (Yang et al., CVPRW 2024),
which combines feature and logit losses, but we add attention transfer as a third signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedFeatureAttentionLoss(nn.Module):
    """Combined feature alignment + attention transfer loss.

    Uses MSE for both components but operates on different representations:
    - Feature: projected student embeddings vs teacher embeddings
    - Attention: head-averaged attention maps

    Args:
        lambda_feature: Weight for feature alignment component.
        lambda_attention: Weight for attention transfer component.
        distill_cls: Include CLS token in feature loss.
        distill_patch: Include patch tokens in feature loss.
        average_heads: Average across attention heads before comparison.
    """

    def __init__(self, lambda_feature=0.1, lambda_attention=0.5,
                 distill_cls=True, distill_patch=True, average_heads=True):
        super().__init__()
        self.lambda_feature = lambda_feature
        self.lambda_attention = lambda_attention
        self.distill_cls = distill_cls
        self.distill_patch = distill_patch
        self.average_heads = average_heads

    def forward(self, student_features, teacher_features,
                student_attention=None, teacher_attention=None):
        """Compute combined loss.

        Args:
            student_features: dict of projected student features per layer
            teacher_features: dict of teacher features per layer
            student_attention: dict of student attention maps per layer
            teacher_attention: dict of teacher attention maps per layer

        Returns:
            Tuple of (total_loss, feature_loss, attention_loss) for logging.
        """
        # Feature alignment loss
        feat_loss = 0.0
        num_feat_layers = len(student_features)
        for layer_name in student_features.keys():
            s_feat = student_features[layer_name]
            t_feat = teacher_features[layer_name]

            if self.distill_cls and self.distill_patch:
                feat_loss += F.mse_loss(s_feat, t_feat)
            elif self.distill_cls:
                feat_loss += F.mse_loss(s_feat[:, 0], t_feat[:, 0])
            elif self.distill_patch:
                feat_loss += F.mse_loss(s_feat[:, 1:], t_feat[:, 1:])

        feat_loss = feat_loss / max(1, num_feat_layers)

        # Attention transfer loss
        attn_loss = 0.0
        if student_attention is not None and teacher_attention is not None:
            num_attn_layers = len(student_attention)
            for layer_name in student_attention.keys():
                s_attn = student_attention[layer_name]
                t_attn = teacher_attention[layer_name]

                if self.average_heads:
                    s_attn = s_attn.mean(dim=1)
                    t_attn = t_attn.mean(dim=1)

                attn_loss += F.mse_loss(s_attn, t_attn)

            attn_loss = attn_loss / max(1, num_attn_layers)

        total = self.lambda_feature * feat_loss + self.lambda_attention * attn_loss
        return total, feat_loss, attn_loss
