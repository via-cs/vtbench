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
            # lightweight attention: learnable vectors 
            self.attn_vectors = nn.Parameter(torch.randn(num_branches, feature_size))
            self.output_size = feature_size

        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, features):
        """
        features: list of [B, d] tensors (one per branch)
        """
        if self.mode == 'concat':
            return torch.cat(features, dim=1)

        elif self.mode == 'weighted_sum':
            # compute per-sample attention scores
            attn_logits = [
                torch.sum(features[i] * self.attn_vectors[i], dim=1, keepdim=True)
                for i in range(self.num_branches)
            ]
            attn_logits = torch.cat(attn_logits, dim=1)        
            attn_weights = F.softmax(attn_logits, dim=1)      
            weighted = sum(attn_weights[:, i].unsqueeze(1) * features[i]
                           for i in range(self.num_branches))
            return weighted
