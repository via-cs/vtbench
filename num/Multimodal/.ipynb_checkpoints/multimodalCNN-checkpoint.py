import torch.nn as nn
import torch
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

class MultimodalDeep2DCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_numerical_features):
        super(MultimodalDeep2DCNN, self).__init__()

       
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = self._get_flatten_size(input_channels)

       
        numerical_method = config["numerical_processing"]["method"]
        if numerical_method == "fc":
            self.numerical_module = self._build_fc_module(num_numerical_features)
            numerical_output_size = 32  # FC output
        elif numerical_method == "transformer":
            self.numerical_module = self._build_transformer_module(num_numerical_features)
            numerical_output_size = config["numerical_processing"]["transformer"]["hidden_dim"]
        else:
            raise ValueError(f"Unsupported numerical processing method: {numerical_method}")

       
        self.fc_combined = nn.Sequential(
            nn.Linear(self.flatten_size + numerical_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, numerical):
        img_features = self.conv_layers(image)
        img_features = img_features.view(img_features.size(0), -1)

        num_features = self.numerical_module(numerical)

        combined_features = torch.cat((img_features, num_features), dim=1)

        output = self.fc_combined(combined_features)
        return output

    def _get_flatten_size(self, input_channels):
        """Calculate the flattened size of the CNN output"""
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)

    def _build_fc_module(self, num_numerical_features):
        """Fully Connected Network for numerical data (Unchanged)"""
        return nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def _build_transformer_module(self, num_numerical_features):
        """Transformer Encoder for numerical data"""
        transformer_config = config["numerical_processing"]["transformer"]
        return TransformerEncoder(
            input_dim=num_numerical_features,
            num_layers=transformer_config["num_layers"],
            num_heads=transformer_config["num_heads"],
            hidden_dim=transformer_config["hidden_dim"],
            dropout=transformer_config["dropout"]
        )


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  

    def forward(self, numerical_data):
        embedded = self.embedding(numerical_data.unsqueeze(1))  
        encoded = self.transformer_encoder(embedded)
        return self.output_layer(encoded.squeeze(1))  
