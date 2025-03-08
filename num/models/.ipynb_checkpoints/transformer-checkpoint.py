import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.3):
        super(TransformerClassifier, self).__init__()

       
        self.embedding = nn.Linear(1, d_model)  # Maps (batch, seq_len, 1) → (batch, seq_len, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x is None:
            raise ValueError("🚨 Error: Received `None` as input!")

        # ✅ Ensure input is 3D (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, input_dim), but got {x.shape}")

        batch_size, seq_len, _ = x.size()

        positional_encoding = self.get_positional_encoding(seq_len, self.embedding.out_features, x.device)

        x = self.embedding(x) + positional_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.fc(x[:, -1, :])
        
        return x

    @staticmethod
    def get_positional_encoding(seq_len, d_model, device):
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)
