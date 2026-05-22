import timm
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224"):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1000
        )

    def forward(self, x, return_features=False):
        if return_features:
            return self.model.forward_features(x)
        return self.model(x)
