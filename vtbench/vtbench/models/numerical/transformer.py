import torch
import torch.nn as nn

class NumericalTransformer(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1, output_dim=64):
        super(NumericalTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B, 1, hidden_dim)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc_out(x)
