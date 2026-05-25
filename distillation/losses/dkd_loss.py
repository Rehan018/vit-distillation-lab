"""
Decoupled Knowledge Distillation (DKD)

Reference: Zhao et al., "Decoupled Knowledge Distillation", CVPR 2022.

Rationale: Standard KD treats all classes equally in the softened distribution.
DKD separates the distillation signal into:
  - Target Class Knowledge Distillation (TCKD): alignment on the ground-truth class
  - Non-Target Class Knowledge Distillation (NCKD): alignment on the dark knowledge
    from non-target classes

This decoupling allows independent weighting of "what the teacher is confident about"
vs "the relative ranking among wrong classes". The latter (NCKD) is where most of the
transferable dark knowledge resides — it encodes inter-class similarity structure that
the student cannot learn from hard labels alone.

Why this approach: Our experiments with Vanilla KD showed that uniform temperature
scaling treats target and non-target logits identically. For fine-grained classification
(Oxford Pets, 37 breeds), the inter-class confusion structure (e.g., Siamese vs Bengal)
is highly informative. DKD allows amplifying this signal by upweighting NCKD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation loss.

    Splits KL divergence into target-class (TCKD) and non-target-class (NCKD)
    components with independent weights, allowing finer control over what
    knowledge the student absorbs.

    Args:
        temperature: Softmax temperature for logit scaling.
        alpha: Weight for hard label CE loss.
        beta: Weight for TCKD component.
        gamma: Weight for NCKD component.
    """

    def __init__(self, temperature=4.0, alpha=0.7, beta=1.0, gamma=8.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets=None):
        if targets is None:
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=1),
                F.softmax(teacher_logits / self.temperature, dim=1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            return kd_loss

        ce = self.ce_loss(student_logits, targets)

        s_soft = F.softmax(student_logits / self.temperature, dim=1)
        t_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        batch_size = student_logits.size(0)
        target_mask = torch.zeros_like(student_logits, dtype=torch.bool)
        target_mask.scatter_(1, targets.unsqueeze(1), True)

        s_target = s_soft[target_mask].view(batch_size, 1)
        t_target = t_soft[target_mask].view(batch_size, 1)


        s_nontarget = s_soft[~target_mask].view(batch_size, -1)
        t_nontarget = t_soft[~target_mask].view(batch_size, -1)

        s_nontarget = s_nontarget / (s_nontarget.sum(dim=1, keepdim=True) + 1e-8)
        t_nontarget = t_nontarget / (t_nontarget.sum(dim=1, keepdim=True) + 1e-8)

        s_binary = torch.cat([s_target, 1.0 - s_target], dim=1)
        t_binary = torch.cat([t_target, 1.0 - t_target], dim=1)
        tckd = F.kl_div(
            torch.log(s_binary + 1e-8),
            t_binary,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        nckd = F.kl_div(
            torch.log(s_nontarget + 1e-8),
            t_nontarget,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        loss = self.alpha * ce + self.beta * tckd + self.gamma * nckd
        return loss
