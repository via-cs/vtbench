import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, mode='concat', feature_size=64, num_branches=4):
        super(FusionModule, self).__init__()
        self.mode = mode
        self.feature_size = feature_size
        self.num_branches = num_branches

        if mode == 'concat':
            self.output_size = feature_size * num_branches
        elif mode == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(num_branches))
            self.output_size = feature_size
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, features):
        if self.mode == 'concat':
            return torch.cat(features, dim=1)
        elif self.mode == 'weighted_sum':
            w = F.softmax(self.weights, dim=0)
            weighted = sum(w[i] * features[i] for i in range(self.num_branches))
            return weighted
