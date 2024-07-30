import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=512, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(src.device)
        src = src.permute(1, 0, 2)  
        memory = self.transformer_encoder(src)
        output = self.fc(memory[-1])
        return output
