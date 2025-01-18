import torch.nn as nn
import torch

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

        self.fc_numerical = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(self.flatten_size + 32, 512),  
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

        num_features = self.fc_numerical(numerical)

        combined_features = torch.cat((img_features, num_features), dim=1)

        output = self.fc_combined(combined_features)
        return output

    def _get_flatten_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)
