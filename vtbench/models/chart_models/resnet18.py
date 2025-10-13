# import torch
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights

# class ResNet18(nn.Module):
#     """ResNet-18 for time series chart classification"""
    
#     def __init__(self, input_channels=3, num_classes=None, pretrained=False):
#         super(ResNet18, self).__init__()
        
#         # Load ResNet-18
#         if pretrained:
#             self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         else:
#             self.model = resnet18(weights=None)
        
#         # Modify first conv layer if input_channels != 3
#         if input_channels != 3:
#             self.model.conv1 = nn.Conv2d(
#                 input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
#             )
        
#         # Get feature dimension from ResNet-18
#         self.feature_dim = self.model.fc.in_features
        
#         # Remove original fc layer
#         self.model.fc = nn.Identity()
        
#         # Add new classification head if num_classes specified
#         if num_classes is not None:
#             self.classifier = nn.Linear(self.feature_dim, num_classes)
#         else:
#             self.classifier = None
    
#     def forward(self, x):
#         # Extract features
#         features = self.model(x)
        
#         # If classifier exists, return logits
#         if self.classifier is not None:
#             return self.classifier(features)
        
#         # Otherwise return features
#         return features
    
#     def get_feature_dim(self):
#         """Return the feature dimension for fusion"""
#         return self.feature_dim


import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    """ResNet-18 for time series chart classification (fully fine-tuned)."""

    def __init__(self, input_channels=3, num_classes=None, pretrained=False):
        super().__init__()

        # Load backbone
        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Conv1 adaptation if channels != 3
        if input_channels != 3:
            old = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                if pretrained:
                    # Reasonable init from RGB weights
                    if input_channels == 1:
                        # average RGB -> single channel
                        w = old.weight.mean(dim=1, keepdim=True)
                        self.model.conv1.weight.copy_(w)
                    elif input_channels > 3:
                        # copy RGB into first 3, average for the rest
                        w = old.weight
                        self.model.conv1.weight[:, :3].copy_(w)
                        mean_w = w.mean(dim=1, keepdim=True)
                        repeat = input_channels - 3
                        self.model.conv1.weight[:, 3:].copy_(mean_w.expand(-1, repeat, -1, -1))
                    else:
                        # input_channels = 2 → take first two RGB channels
                        w = old.weight
                        self.model.conv1.weight[:, :input_channels].copy_(w[:, :input_channels])
                else:
                    # Kaiming init is fine for scratch
                    nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Replace FC with Identity; we add our own head
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.classifier = nn.Linear(self.feature_dim, num_classes) if num_classes is not None else None

    def forward(self, x):
        feats = self.model(x)  # (B, feature_dim)
        return self.classifier(feats) if self.classifier is not None else feats

    def get_feature_dim(self):
        return self.feature_dim
