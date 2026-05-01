import torch
import torch.nn as nn

class NumericalFCN(nn.Module):
    def __init__(self, input_dim=96, output_dim=64):
        super(NumericalFCN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc_layers(x)
