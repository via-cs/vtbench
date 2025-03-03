import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(FCN, self).__init__()
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

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Linear(input_dim, config['hidden_dim'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'], 
            nhead=config['num_heads'], 
            dim_feedforward=config['hidden_dim'] * 2, 
            dropout=config['dropout'],
            batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        output_size = config.get('output_size', 256) 
        self.fc_out = nn.Linear(config['hidden_dim'], output_size)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc_out(x)


class Simple2DCNN(nn.Module):
    def __init__(self, input_channels, num_classes=None):
        super(Simple2DCNN, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten_size = self._get_flatten_size(input_channels)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _get_flatten_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)



class Deep2DCNN(nn.Module):
    def __init__(self, input_channels):
        super(Deep2DCNN, self).__init__()
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
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x



class MultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_features, config):
        super(MultiBranchCNN, self).__init__()
        
        self.fusion_method = config['fusion_method']
        self.numerical_method = config['numerical_processing']['method']
        
        if config['architecture'] == "SimpleMultiBranchCNN":
            CNNModel = Simple2DCNN
            self.feature_size = 64
        else:
            CNNModel = Deep2DCNN
            self.feature_size = 256
        
        self.bar_branch = CNNModel(input_channels)
        self.line_branch = CNNModel(input_channels)
        self.area_branch = CNNModel(input_channels)
        self.scatter_branch = CNNModel(input_channels)

        if self.numerical_method == 'fcn':
            self.numerical_branch = FCN(
                input_dim=num_features,
                output_dim=self.feature_size  # Match CNN feature size
            )
            self.numerical_output_size = self.feature_size
        elif self.numerical_method == 'transformer':
            transformer_config = config['numerical_processing']['transformer']
            transformer_config['output_size'] = self.feature_size  
            
            self.numerical_branch = TransformerEncoder(
                input_dim=num_features,
                config=transformer_config
            )
            self.numerical_output_size = self.feature_size
        else:
            self.numerical_branch = None
            self.numerical_output_size = 0

        # Calculating fusion size
        if self.fusion_method == 'concatenation':
            self.fusion_size = self.feature_size * 4
            self.classifier_input_size = self.fusion_size + self.numerical_output_size
        else:  # weighted_sum
            self.fusion_size = self.feature_size
            self.classifier_input_size = self.feature_size
            
        # Initialize fusion weights if using weighted sum
        if self.fusion_method == 'weighted_sum':
            num_weights = 5 if self.numerical_branch is not None else 4
            self.fusion_weights = nn.Parameter(torch.ones(num_weights))
        
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, bar_img, line_img, area_img, scatter_img, numerical_data=None):
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)
        
         # Process numerical data if available
        if self.numerical_branch is not None and numerical_data is not None:
            numerical_features = self.numerical_branch(numerical_data)
        else:
            numerical_features = None
        
        # Feature fusion
        if self.fusion_method == 'concatenation':
            # Concatenate CNN features
            combined = torch.cat([
                bar_features,
                line_features,
                area_features,
                scatter_features
            ], dim=1)
            
            # Add numerical features if present
            if numerical_features is not None:
                combined = torch.cat([combined, numerical_features], dim=1)
                
        else:  # weighted_sum
            # Normalize weights
            weights = F.softmax(self.fusion_weights, dim=0)
            
            # Combine CNN features with weights
            combined = (
                weights[0] * bar_features +
                weights[1] * line_features +
                weights[2] * area_features +
                weights[3] * scatter_features
            )
            
            # Add numerical features if present
            if numerical_features is not None:
                combined = combined + weights[4] * numerical_features
        
        # Classification
        output = self.classifier(combined)
        return output