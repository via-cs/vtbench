import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(FeatureExtractor, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, stride=2)
        self.res3 = ResidualBlock(64, 64)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x

class MultiBranchCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultiBranchCNN, self).__init__()
        
        # Feature extractors for each chart type
        self.bar_branch = FeatureExtractor(input_channels)
        self.line_branch = FeatureExtractor(input_channels)
        self.area_branch = FeatureExtractor(input_channels)
        self.scatter_branch = FeatureExtractor(input_channels)
        
        self.feature_size = 64  # Matches the output of FeatureExtractor
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),  # Residual connection will be added in forward pass
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, bar_img, line_img, area_img, scatter_img):
        # Extract features from each branch
        bar_features = self.bar_branch(bar_img)
        line_features = self.line_branch(line_img)
        area_features = self.area_branch(area_img)
        scatter_features = self.scatter_branch(scatter_img)
        
        # Normalize features
        features = [
            F.normalize(bar_features, p=2, dim=1),
            F.normalize(line_features, p=2, dim=1),
            F.normalize(area_features, p=2, dim=1),
            F.normalize(scatter_features, p=2, dim=1)
        ]
        
        # Average the normalized features
        combined = torch.cat(features, dim=1)
        
        # Apply classifier with residual connection
        x = self.classifier[:4](combined)  # First dense block
        identity = x
        
        x = self.classifier[4:8](x)  # Second dense block
        x = x + identity  # Residual connection
        
        x = self.classifier[8:](x)  # Final layers
        
        return x