import torch.nn as nn

class LinearClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
      
        return self.head(x)
