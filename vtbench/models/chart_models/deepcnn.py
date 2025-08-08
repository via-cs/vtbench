import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, input_channels, num_classes=None, input_size=64):
        super(DeepCNN, self).__init__()
        self.input_size = input_size
        
        # Optimized architecture with bias=False and inplace operations
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dynamically calculate flatten size
        self.flatten_size = self._get_flatten_size(input_channels)
        
        # Feature extractor with BatchNorm
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)  # Keep bias in final layer
            )
        else:
            self.classifier = nn.Identity()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
    def _get_flatten_size(self, input_channels):
        # Use actual input_size instead of fixed 64x64
        dummy_input = torch.zeros(1, input_channels, self.input_size, self.input_size)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)
