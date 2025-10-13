import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, input_channels, num_classes=None, input_size=64):
        super(DeepCNN, self).__init__()
        self.input_size = input_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # --- Minimal fix: use LazyLinear so the flatten size is inferred at first forward ---
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512, bias=False),   # <- was nn.Linear(self.flatten_size, 512, ...)
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        else:
            self.classifier = nn.Identity()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
    # kept for backward compatibility; no longer used
    def _get_flatten_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, self.input_size, self.input_size)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)
