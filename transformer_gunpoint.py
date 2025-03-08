import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        self.transformer = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=0,
            dim_feedforward=512,
            dropout=0.19827646166104682,
            batch_first=True
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, _ = x.size()
        positional_encoding = self.get_positional_encoding(seq_len, self.embedding.out_features, x.device)
        x = self.embedding(x) + positional_encoding[:, :seq_len, :]
        x = self.transformer.encoder(x)
        return self.fc(x[:, -1, :])

    @staticmethod
    def get_positional_encoding(seq_len, d_model, device):
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)
