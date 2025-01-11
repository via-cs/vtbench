import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(-1)  

        batch_size, seq_len, _ = x.size()
        d_model = self.embedding.out_features

        positional_encoding = self.get_positional_encoding(seq_len, d_model, x.device)
        x = self.embedding(x) + positional_encoding[:, :seq_len, :]
        x = self.transformer.encoder(x)
        x = x[:, -1, :]  
        return self.fc(x)

    @staticmethod
    def get_positional_encoding(seq_len, d_model, device):
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)
