
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, patch_size=16, img_size=224, hidden_dim=256):
        super().__init__()
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.img_size = img_size

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, features):
        """features: [B, N, D] where N = 1 + num_patches"""
        patch_tokens = features[:, 1:, :]
        B, N, D = patch_tokens.shape
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, D, self.num_patches_h, self.num_patches_w)
        logits = self.decoder(patch_tokens)
        logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return logits
