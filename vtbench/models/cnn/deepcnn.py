import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self, input_channels, num_classes=None):
        super(DeepCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten_size = self._get_flatten_size(input_channels)

        # If num_classes is None, act as a feature extractor
        if num_classes is None:
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flatten_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU()
                # Output = 256-dim feature
            )
        else:
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.flatten_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)  # Final classification layer
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _get_flatten_size(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 64, 64)
        x = self.conv_layers(dummy_input)
        return x.view(1, -1).size(1)
